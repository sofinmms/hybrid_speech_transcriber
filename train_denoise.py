import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === параметры ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
EPOCHS = 100
BATCH_SIZE = 32
MAX_AUDIO_LENGTH = 2 * SR  # 2 секунды

# === исправленная U-Net архитектура ===
def build_unet_denoise_model(input_shape):
    """U-Net архитектура с правильными размерами для конкатенации"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    # Block 1
    conv1 = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv1D(32, 5, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling1D(2, padding='same')(conv1)  # (None, 129, 32)
    
    # Block 2
    conv2 = layers.Conv1D(64, 5, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv1D(64, 5, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling1D(2, padding='same')(conv2)  # (None, 65, 64)
    
    # Block 3 (bottleneck)
    conv3 = layers.Conv1D(128, 5, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv1D(128, 5, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    
    # Decoder
    # Block 4
    up4 = layers.UpSampling1D(2)(conv3)  # (None, 130, 128)
    # Обрезаем или дополняем для совпадения размеров
    up4 = layers.Cropping1D((0, 1))(up4) if up4.shape[1] > conv2.shape[1] else up4  # (None, 129, 128)
    up4 = layers.ZeroPadding1D((0, conv2.shape[1] - up4.shape[1]))(up4) if up4.shape[1] < conv2.shape[1] else up4
    
    concat4 = layers.Concatenate()([up4, conv2])
    conv4 = layers.Conv1D(64, 5, activation='relu', padding='same')(concat4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv1D(64, 5, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    
    # Block 5
    up5 = layers.UpSampling1D(2)(conv4)  # (None, 130, 64)
    # Обрезаем или дополняем для совпадения размеров
    up5 = layers.Cropping1D((0, 1))(up5) if up5.shape[1] > conv1.shape[1] else up5  # (None, 129, 64)
    up5 = layers.ZeroPadding1D((0, conv1.shape[1] - up5.shape[1]))(up5) if up5.shape[1] < conv1.shape[1] else up5
    
    concat5 = layers.Concatenate()([up5, conv1])
    conv5 = layers.Conv1D(32, 5, activation='relu', padding='same')(concat5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv1D(32, 5, activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    
    # Output
    outputs = layers.Conv1D(2, 3, activation='tanh', padding='same')(conv5)
    
    model = models.Model(inputs, outputs)
    return model

# === альтернативная простая архитектура ===
def build_simple_denoise_model(input_shape):
    """Простая, но эффективная архитектура без skip connections"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Первые слои с большим количеством фильтров
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Conv1D(512, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Conv1D(512, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Выходной слой
        layers.Conv1D(2, 3, activation='tanh', padding='same')
    ])
    return model

# === функция потерь для комплексных чисел ===
def complex_mse_loss(y_true, y_pred):
    """MSE loss для реальной и мнимой частей"""
    real_loss = tf.reduce_mean(tf.square(y_true[..., 0] - y_pred[..., 0]))
    imag_loss = tf.reduce_mean(tf.square(y_true[..., 1] - y_pred[..., 1]))
    return real_loss + imag_loss

# === утилиты для обработки аудио ===
def audio_to_stft_complex(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Преобразует аудио в комплексный STFT"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    real = stft.real.T  # (time, freq)
    imag = stft.imag.T  # (time, freq)
    return real, imag

def stft_complex_to_audio(real, imag, hop_length=HOP_LENGTH):
    """Преобразует реальную и мнимую части обратно в аудио"""
    stft = (real + 1j * imag).T  # Транспонируем обратно
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

def preprocess_audio(audio, target_length=MAX_AUDIO_LENGTH):
    """Предобработка аудио: обрезка/дополнение и нормализация"""
    if len(audio) > target_length:
        # Случайная обрезка
        start = np.random.randint(0, len(audio) - target_length)
        audio = audio[start:start + target_length]
    else:
        # Дополнение нулями
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Нормализация по пиковому значению
    max_val = np.max(np.abs(audio)) + 1e-8
    audio = audio / max_val
    
    return audio

# === подготовка данных ===
def prepare_dataset(clean_files, noisy_files_dir, num_samples=30000):
    """Подготавливает dataset из парных clean/noisy frames"""
    X_real, X_imag, Y_real, Y_imag = [], [], [], []
    
    # Создаем список пар файлов
    file_pairs = []
    for clean_file in clean_files:
        filename = os.path.basename(clean_file)
        noisy_file = os.path.join(noisy_files_dir, filename)
        
        if os.path.exists(noisy_file):
            file_pairs.append((clean_file, noisy_file))
    
    print(f"Найдено {len(file_pairs)} парных файлов")
    
    if not file_pairs:
        raise ValueError("Не найдено парных файлов для обучения!")
    
    random.shuffle(file_pairs)
    
    for i, (clean_file, noisy_file) in enumerate(file_pairs):
        if len(X_real) >= num_samples:
            break
            
        if i % 100 == 0:
            print(f"Обработано {i} файлов, собрано {len(X_real)} frames")
            
        try:
            # Загружаем аудио
            clean_audio, _ = librosa.load(clean_file, sr=SR)
            noisy_audio, _ = librosa.load(noisy_file, sr=SR)
            
            # Предобработка
            clean_audio = preprocess_audio(clean_audio)
            noisy_audio = preprocess_audio(noisy_audio)
            
            # Преобразуем в STFT
            clean_real, clean_imag = audio_to_stft_complex(clean_audio)
            noisy_real, noisy_imag = audio_to_stft_complex(noisy_audio)
            
            # Убедимся, что размеры совпадают
            min_frames = min(len(clean_real), len(noisy_real))
            clean_real = clean_real[:min_frames]
            clean_imag = clean_imag[:min_frames]
            noisy_real = noisy_real[:min_frames]
            noisy_imag = noisy_imag[:min_frames]
            
            # Добавляем frames в dataset
            for j in range(0, min_frames, 5):  # Берем каждый 5-й frame
                if len(X_real) >= num_samples:
                    break
                    
                X_real.append(noisy_real[j])
                X_imag.append(noisy_imag[j])
                Y_real.append(clean_real[j])
                Y_imag.append(clean_imag[j])
                    
        except Exception as e:
            print(f"Ошибка обработки файлов {clean_file}, {noisy_file}: {e}")
            continue
    
    # Преобразуем в numpy arrays
    X_real = np.array(X_real)
    X_imag = np.array(X_imag)
    Y_real = np.array(Y_real)
    Y_imag = np.array(Y_imag)
    
    # Глобальная нормализация
    max_val = max(np.max(np.abs(X_real)), np.max(np.abs(X_imag)),
                 np.max(np.abs(Y_real)), np.max(np.abs(Y_imag))) + 1e-8
    
    X_real /= max_val
    X_imag /= max_val
    Y_real /= max_val
    Y_imag /= max_val
    
    # Объединяем реальную и мнимую части
    X = np.stack([X_real, X_imag], axis=-1)
    Y = np.stack([Y_real, Y_imag], axis=-1)
    
    print(f"Размер dataset: X {X.shape}, Y {Y.shape}")
    
    return X, Y

# === функция для тестирования ===
def test_model(model, test_files, output_dir="test_results"):
    """Тестирует модель на нескольких файлах с визуализацией"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (clean_file, noisy_file) in enumerate(test_files[:3]):
        try:
            print(f"Тестирование файла {i+1}/{min(3, len(test_files))}")
            
            # Загружаем аудио
            clean_audio, _ = librosa.load(clean_file, sr=SR)
            noisy_audio, _ = librosa.load(noisy_file, sr=SR)
            
            # Предобработка
            clean_audio = preprocess_audio(clean_audio)
            noisy_audio = preprocess_audio(noisy_audio)
            
            # Преобразуем в STFT
            noisy_real, noisy_imag = audio_to_stft_complex(noisy_audio)
            
            # Обрабатываем батчами для эффективности
            enhanced_real, enhanced_imag = [], []
            batch_size = 32
            
            for start_idx in range(0, len(noisy_real), batch_size):
                end_idx = min(start_idx + batch_size, len(noisy_real))
                batch_frames = []
                
                for j in range(start_idx, end_idx):
                    input_frame = np.stack([noisy_real[j], noisy_imag[j]], axis=-1)
                    batch_frames.append(input_frame)
                
                batch_frames = np.array(batch_frames)
                predictions = model.predict(batch_frames, verbose=0)
                
                for pred in predictions:
                    enhanced_real.append(pred[:, 0])
                    enhanced_imag.append(pred[:, 1])
            
            # Обратное преобразование
            enhanced_audio = stft_complex_to_audio(np.array(enhanced_real), 
                                                 np.array(enhanced_imag))
            
            # Обрезаем до исходной длины
            min_length = min(len(clean_audio), len(enhanced_audio))
            clean_audio = clean_audio[:min_length]
            enhanced_audio = enhanced_audio[:min_length]
            
            # Сохраняем результаты
            sf.write(os.path.join(output_dir, f"original_{i}.wav"), clean_audio, SR)
            sf.write(os.path.join(output_dir, f"noisy_{i}.wav"), noisy_audio[:min_length], SR)
            sf.write(os.path.join(output_dir, f"enhanced_{i}.wav"), enhanced_audio, SR)
            
            # Визуализация спектрограмм
            plt.figure(figsize=(15, 12))
            
            # Original
            plt.subplot(3, 1, 1)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Original Clean Audio')
            plt.colorbar(format='%+2.0f dB')
            
            # Noisy
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio[:min_length], n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Noisy Audio')
            plt.colorbar(format='%+2.0f dB')
            
            # Enhanced
            plt.subplot(3, 1, 3)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Enhanced Audio')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spectrogram_{i}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Сохранены результаты для файла {i}")
            
        except Exception as e:
            print(f"Ошибка при тестировании файла {clean_file}: {e}")
            continue

# === основной код ===
if __name__ == "__main__":
    # Игнорируем предупреждения
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Создаем директории
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    
    # Собираем файлы
    clean_files = glob.glob("data/clean_speech/*.wav")
    noisy_files_dir = "data/noise/"
    
    if not clean_files:
        raise RuntimeError("❌ Не найдено файлов в data/clean_speech/")
    if not os.path.exists(noisy_files_dir):
        raise RuntimeError("❌ Не найдена папка data/noise/")

    print(f"Найдено {len(clean_files)} чистых файлов")
    
    # Разделяем на train/test
    train_files, test_files = train_test_split(clean_files, test_size=0.1, random_state=42)
    test_files = [(f, os.path.join(noisy_files_dir, os.path.basename(f))) for f in test_files]

    # Создаем модель (используем простую архитектуру для надежности)
    input_shape = (N_FFT//2 + 1, 2)  # (257, 2)
    
    # Выбираем архитектуру
    print("Попытка создания U-Net архитектуры...")
    model = build_unet_denoise_model(input_shape)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                 loss=complex_mse_loss,
                 metrics=['mae'])
    model.summary()

    # Подготавливаем данные
    print("Подготовка данных...")
    X_train, Y_train = prepare_dataset(train_files, noisy_files_dir, num_samples=30000)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint("models/best_model.weights.h5", monitor='val_loss', 
                       save_best_only=True, save_weights_only=True, verbose=1)
    ]

    # Обучаем модель
    print("Начало обучения...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.15,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )

    # Сохраняем модель
    model.save_weights("models/final_denoise_model.weights.h5")
    print("✅ Модель обучена и сохранена")

    # Тестируем
    print("Тестирование модели...")
    test_model(model, test_files)
    
    # Сохраняем историю обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
    
    plt.savefig("models/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Обучение завершено! Результаты в test_results/ и models/")