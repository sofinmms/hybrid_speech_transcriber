# test_denoise_fixed.py
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models

# === параметры ===
SR = 16000
N_FFT = 512
HOP = 128

# === модель ===
def build_enhanced_denoise_model(n_fft=512):
    model = models.Sequential([
        layers.Input(shape=(n_fft//2 + 1, 2)),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv1D(2, 3, activation='tanh', padding='same')
    ])
    return model

def process_audio_with_model_fixed(audio, model, n_fft=N_FFT, hop=HOP):
    """Исправленная обработка аудио"""
    # Получаем STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    stft_real = stft.real
    stft_imag = stft.imag
    
    # Нормализуем всю STFT матрицу целиком
    max_val = np.max(np.abs(stft))
    if max_val > 0:
        stft_real_normalized = stft_real / max_val
        stft_imag_normalized = stft_imag / max_val
    else:
        stft_real_normalized = stft_real
        stft_imag_normalized = stft_imag
    
    # Подготавливаем вход для модели
    frames = np.stack([stft_real_normalized, stft_imag_normalized], axis=-1)
    frames = np.transpose(frames, (1, 0, 2))  # (time, freq, 2)
    
    # Обрабатываем все кадры сразу (батчем)
    processed_frames = model.predict(frames, verbose=1)
    
    # Денормализуем
    if max_val > 0:
        processed_frames = processed_frames * max_val
    
    # Собираем обратно в STFT матрицу
    processed_frames = np.transpose(processed_frames, (1, 0, 2))  # (freq, time, 2)
    stft_clean = processed_frames[..., 0] + 1j * processed_frames[..., 1]
    
    return stft_clean

def stft_to_audio(stft_matrix, hop=HOP):
    return librosa.istft(stft_matrix, hop_length=hop)

# === запуск ===
if __name__ == "__main__":
    # загружаем зашумленное аудио
    print("Загрузка аудио...")
    noisy, _ = librosa.load("data/noise/p287_417.wav", sr=SR)
    
    # строим модель и грузим веса
    print("Загрузка модели...")
    model = build_enhanced_denoise_model(N_FFT)
    model.load_weights("denoise_model.weights.h5")
    
    # обрабатываем аудио
    print("Обработка аудио...")
    stft_clean = process_audio_with_model_fixed(noisy, model)
    
    # преобразуем в аудио
    print("Синтез аудио...")
    audio_clean = stft_to_audio(stft_clean)
    
    # сохраняем
    sf.write("denoised_fixed.wav", audio_clean, SR)
    print("✅ Результат сохранён в denoised_fixed.wav")