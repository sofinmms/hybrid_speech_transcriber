import numpy as np
import librosa
import scipy.ndimage
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import webrtcvad
from scipy import signal

class HybridSpeechEnhancer:
    def __init__(self, sr, n_fft=512, hop_length=320, win_length=480,
                 frame_duration=0.02, vad_aggressiveness=3, online_learning=True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.window = np.hanning(self.win)
        self.frame_duration = frame_duration
        self.online_learning = online_learning
        
        # Инициализация WebRTC VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_aggressiveness)
        
        # Буфер для накопления кадров
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_size = int(sr * 0.1)
        
        # Буферы для онлайн-обучения
        self.clean_buffer = deque(maxlen=1000)
        self.noisy_buffer = deque(maxlen=1000)
        self.learning_interval = 50
        self.frame_counter = 0
        
        # Параметры гибридного подавления
        self.noise_estimate = None
        self.noise_floor = 0.01
        self.spec_subtraction_alpha = 1.5  # Коэффициент спектрального вычитания
        self.spec_subtraction_beta = 0.1   # Коэффициент для защиты речи
        self.min_snr_for_learning = 15
        self.learning_confidence_threshold = 0.7
        self.max_learning_epochs = 100
        self.learning_epoch_counter = 0
        
        # Инициализация модели шумоподавления
        self.denoise_model = self._build_denoise_model()

        try:
            self.denoise_model.load_weights('denoise_model.weights.h5')
        except:
            print("Предобученные веса не найдены. Инициализированы случайные веса.")
        
        # Компиляция модели
        self.denoise_model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
        
        # Состояние VAD
        self.speech_state = False
        self.speech_history = deque(maxlen=5)
        self.consecutive_speech_frames = 0
    

    def _estimate_snr(self, mag):
        """Оценивает SNR сигнала."""
        # Простая оценка SNR на основе энергии
        energy = np.mean(mag**2)
        noise_energy = np.mean(self.noise_estimate**2) if self.noise_estimate is not None else 1e-8
        snr_db = 10 * np.log10(energy / (noise_energy + 1e-8))
        return snr_db
    
    def _estimate_confidence(self, stft_features):
        """Оценивает уверенность в качестве оценки чистого сигнала."""
        # Можно реализовать различные метрики уверенности
        mag = np.abs(stft_features)
        spectral_flatness = np.exp(np.mean(np.log(mag + 1e-8))) / np.mean(mag)
        
        # Низкая spectral flatness обычно указывает на наличие речи
        confidence = 1 - spectral_flatness
        return np.clip(confidence, 0, 1)
    
    def _get_model_prediction(self, features):
        """Получает предсказание модели для features."""
        # Нормализация
        input_mean = np.mean(features, axis=0, keepdims=True)
        input_std = np.std(features, axis=0, keepdims=True) + 1e-8
        features_normalized = (features - input_mean) / input_std
        
        # Предсказание
        prediction = self.denoise_model.predict(features_normalized[np.newaxis, ...], verbose=0)[0]
        
        # Денормализация
        return prediction * input_std + input_mean
    
    def _train_on_batch(self):
        """Обучает модель на накопленном батче."""
        batch_size = min(32, len(self.clean_buffer))
        clean_batch = random.sample(list(self.clean_buffer), batch_size)
        noisy_batch = random.sample(list(self.noisy_buffer), batch_size)
        
        clean_array = np.array(clean_batch)
        noisy_array = np.array(noisy_batch)
        
        # Используем очень маленький learning rate для осторожного обучения
        original_lr = tf.keras.backend.get_value(self.denoise_model.optimizer.lr)
        tf.keras.backend.set_value(self.denoise_model.optimizer.lr, original_lr * 0.1)
        
        self.denoise_model.train_on_batch(noisy_array, clean_array)
        
        # Восстанавливаем learning rate
        tf.keras.backend.set_value(self.denoise_model.optimizer.lr, original_lr)
    
    def _safe_online_learning_step(self, stft_features, is_speech):
        """Безопасный шаг онлайн-обучения с проверками."""
        if not self.online_learning:
            return
            
        if not self._should_learn(stft_features, is_speech):
            return
            
        # Только для высококачественных речевых фреймов
        real_part = np.real(stft_features)
        imag_part = np.imag(stft_features)
        noisy_features = np.stack([real_part[:, 0], imag_part[:, 0]], axis=-1)
        
        # Используем текущую модель для получения "чистого" сигнала
        # Это безопаснее, чем _estimate_clean_signal
        clean_features = self._get_model_prediction(noisy_features)
        
        # Добавляем в буфер
        self.noisy_buffer.append(noisy_features)
        self.clean_buffer.append(clean_features)
        
        # Обучаем только если накопилось достаточно качественных примеров
        if len(self.clean_buffer) >= 32:
            self._train_on_batch()
            self.learning_epoch_counter += 1

    def _should_learn(self, stft_features, is_speech):
        """Определяет, безопасно ли проводить онлайн-обучение."""
        if not is_speech:
            return False  # Не обучаем на шуме
            
        # Проверяем качество сигнала
        mag = np.abs(stft_features)
        snr = self._estimate_snr(mag)
        
        # Проверяем уверенность в оценке чистого сигнала
        confidence = self._estimate_confidence(stft_features)
        
        return (snr > self.min_snr_for_learning and 
                confidence > self.learning_confidence_threshold and
                self.learning_epoch_counter < self.max_learning_epochs)
        
    def _build_denoise_model(self):
        """Создает модель для шумоподавления."""
        model = models.Sequential([
            layers.Input(shape=(self.n_fft//2 + 1, 2)),
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
    
    def _vad_detection_webrtc(self, audio_frame):
        """Определяет наличие речи с помощью WebRTC VAD."""
        audio_int16 = (audio_frame * 32767).astype(np.int16)
        
        try:
            is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sr)
        except:
            energy = np.mean(audio_frame**2)
            is_speech = energy > 0.001
            
        self.speech_history.append(is_speech)
        speech_count = sum(self.speech_history)
        
        # Обновляем счетчик последовательных речевых фреймов
        if is_speech:
            self.consecutive_speech_frames += 1
        else:
            self.consecutive_speech_frames = 0
            
        return speech_count >= len(self.speech_history) // 2
    
    def _extract_features(self, audio_chunk):
        """Извлекает признаки из аудиочанка."""
        S = librosa.stft(audio_chunk, n_fft=self.n_fft, hop_length=self.hop,
                         win_length=self.win, window=self.window)
        return S
    
    def _update_noise_estimate(self, stft_features, is_speech):
        """Обновляет оценку шумового спектра."""
        mag = np.abs(stft_features)
        
        if self.noise_estimate is None:
            self.noise_estimate = mag
            return
        
        # Адаптивная оценка шума (обновляем только в отсутствие речи)
        if not is_speech or self.consecutive_speech_frames < 3:
            alpha = 0.9  # Коэффициент сглаживания
            self.noise_estimate = alpha * self.noise_estimate + (1 - alpha) * mag
    
    def _spectral_subtraction(self, stft_features):
        """Применяет спектральное вычитание."""
        mag = np.abs(stft_features)
        phase = np.angle(stft_features)
        
        # Защищаем речь от чрезмерного подавления
        speech_mask = np.maximum(0, 1 - self.spec_subtraction_beta * self.noise_estimate / (mag + 1e-8))
        speech_mask = np.minimum(1, speech_mask)
        
        # Спектральное вычитание
        mag_clean = np.maximum(0, mag - self.spec_subtraction_alpha * self.noise_estimate)
        mag_clean = mag_clean * speech_mask  # Применяем защиту речи
        
        return mag_clean * np.exp(1j * phase)
    
    def _wiener_filter(self, stft_features):
        """Применяет адаптивный фильтр Винера."""
        mag = np.abs(stft_features)
        phase = np.angle(stft_features)
        
        # Оценка отношения сигнал/шум
        snr_estimate = np.maximum(0, mag**2 / (self.noise_estimate**2 + 1e-8) - 1)
        wiener_gain = snr_estimate / (snr_estimate + 1)
        
        # Применяем фильтр
        mag_clean = mag * wiener_gain
        
        return mag_clean * np.exp(1j * phase)
    
    def _neural_denoise(self, stft_features):
        """Применяет нейросетевое шумоподавление."""
        real_part = np.real(stft_features)
        imag_part = np.imag(stft_features)
        
        output_real = np.zeros_like(real_part)
        output_imag = np.zeros_like(imag_part)
        
        for t in range(real_part.shape[1]):
            input_frame = np.stack([real_part[:, t], imag_part[:, t]], axis=-1)
            
            # Нормализация
            input_mean = np.mean(input_frame, axis=0, keepdims=True)
            input_std = np.std(input_frame, axis=0, keepdims=True) + 1e-8
            input_normalized = (input_frame - input_mean) / input_std
            
            # Применение модели
            output_frame = self.denoise_model.predict(input_normalized[np.newaxis, ...], verbose=0)[0]
            
            # Денормализация
            output_denormalized = output_frame * input_std + input_mean
            
            output_real[:, t] = output_denormalized[:, 0]
            output_imag[:, t] = output_denormalized[:, 1]
        
        stft_clean = output_real + 1j * output_imag
        return stft_clean
    
    def _hybrid_denoise(self, stft_features, is_speech):
        """Гибридное шумоподавление."""
        # 1. Обновляем оценку шума
        self._update_noise_estimate(stft_features, is_speech)
        
        # 2. Применяем спектральное вычитание как предобработку
        stft_preprocessed = self._spectral_subtraction(stft_features)
        
        # 3. Нейросетевое шумоподавление
        # stft_neural = self._neural_denoise(stft_preprocessed)
        
        # 4. Фильтр Винера как постобработка
        stft_final = self._wiener_filter(stft_preprocessed)
        
        return stft_final
    
    def _estimate_clean_signal(self, noisy_stft):
        """Оценивает чистый сигнал для обучения."""
        mag = np.abs(noisy_stft)
        phase = np.angle(noisy_stft)
        
        # Более точная оценка чистого сигнала
        if self.noise_estimate is not None:
            # Используем адаптивный порог на основе оценки шума
            threshold = np.percentile(self.noise_estimate, 80)
            clean_mag = np.where(mag > threshold, mag, mag * 0.05)
        else:
            threshold = np.percentile(mag, 70)
            clean_mag = np.where(mag > threshold, mag, mag * 0.1)
        
        clean_stft = clean_mag * np.exp(1j * phase)
        
        real_part = np.real(clean_stft)
        imag_part = np.imag(clean_stft)
        
        clean_features = np.stack([real_part[:, 0], imag_part[:, 0]], axis=-1)
        
        return clean_features
    

    def _process_single_frame(self, frame):
        """Обрабатывает один полный фрейм аудио."""
        # Проверяем размер фрейма
        if len(frame) != self.win:
            frame = np.pad(frame, (0, self.win - len(frame)))
        
        # Определяем наличие речи
        is_speech = self._vad_detection_webrtc(frame)
        
        # Извлекаем STFT features
        stft = self._extract_features(frame)
        
        # Применяем гибридное шумоподавление
        if is_speech:
            stft_clean = self._hybrid_denoise(stft, is_speech)
        else:
            if self.noise_estimate is not None:
                mag = np.abs(stft)
                mag_clean = np.maximum(0, mag - 1.8 * self.noise_estimate)
                stft_clean = mag_clean * np.exp(1j * np.angle(stft))
            else:
                mag = np.abs(stft)
                mag_clean = mag * 0.05
                stft_clean = mag_clean * np.exp(1j * np.angle(stft))
        
        audio_clean = librosa.istft(stft_clean, hop_length=self.hop,
                                win_length=self.win, window=self.window)
        
        # Периодическое обучение
        if self.frame_counter % self.learning_interval == 0:
            self._safe_online_learning_step(stft, is_speech)
        
        self.frame_counter += 1
        
        return audio_clean
    
    
    def process_chunk_realtime(self, chunk):
        """Оптимизированная обработка для реального времени."""
        # Для real-time используем hop_length = chunk_size
        if len(chunk) != self.hop:
            chunk = chunk[:self.hop]  # Обрезаем до нужного размера
        
        # Дополняем до размера окна используя предыдущие samples
        if not hasattr(self, 'prev_samples'):
            self.prev_samples = np.zeros(self.win - self.hop)
        
        full_frame = np.concatenate([self.prev_samples, chunk])
        
        # Сохраняем для следующего вызова
        self.prev_samples = full_frame[self.hop:]
        
        # Обрабатываем полный фрейм
        processed_frame = self._process_single_frame(full_frame)
        
        # Возвращаем только соответствующую часть
        return processed_frame[-self.hop:]
    
    def enable_online_learning(self, enabled=True):
        """Включает/выключает онлайн-обучение."""
        self.online_learning = enabled
    
    def set_hybrid_parameters(self, alpha=1.5, beta=0.1, noise_floor=0.01):
        """Устанавливает параметры гибридного подавления."""
        self.spec_subtraction_alpha = alpha  # Агрессивность спектрального вычитания
        self.spec_subtraction_beta = beta    # Защита речи
        self.noise_floor = noise_floor       # Минимальный уровень шума
    
    def save_models(self, denoise_path='denoise_model.weights.h5'):
        """Сохраняет веса модели шумоподавления."""
        self.denoise_model.save_weights(denoise_path)
        print("Модель сохранена")