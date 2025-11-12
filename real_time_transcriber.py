import queue
import threading
import time
import json
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import webrtcvad
from hybrid_speech_enhancer import HybridSpeechEnhancer
from translator import Translator
from vosk import Model, KaldiRecognizer, SetLogLevel
from collections import deque

class RealTimeTranscriber:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sample_rate = 16000
        self.channels = 1
        self.frame_duration_ms = 20
        self.noise_reduction = True
        self.translator = Translator(source_lang='ru', target_lang='en')
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue(maxsize=200)
        self.segment_q = queue.Queue()
        self.combined_noise = None
        self.hybrid_speech_enhancer = HybridSpeechEnhancer(sr = self.sample_rate, vad_aggressiveness=0)
        self.hybrid_speech_enhancer.set_hybrid_parameters(
            alpha=1.7,    # Более агрессивное подавление
            beta=0.2,     # Сильная защита речи
            noise_floor=0.005
        )
        self.hybrid_speech_enhancer.enable_online_learning(False)
        # self.vad = Vad(sample_rate = self.sample_rate, frame_duration_ms = self.frame_duration_ms, energy_threshold = 0.6)
        self.vad = webrtcvad.Vad(3)
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(False)

        self.initialized_noise_profile = False

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("sounddevice status:", status, file=sys.stderr)
        if self.channels > 1:
            indata = indata.mean(axis=1)
        else:
            indata = indata[:, 0]

        try:
            self.audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def producer_thread(self):
        frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        sample_buf = deque()
        while not self.stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            sample_buf.extend(frame.tolist())
            while len(sample_buf) >= frame_samples:
                chunk = np.array([sample_buf.popleft() for _ in range(frame_samples)], dtype=np.float32)

                if self.noise_reduction:
                    clean = self.hybrid_speech_enhancer.process_chunk_realtime(chunk)
                    # print(clean)
                else:
                    clean = chunk
            
                ts = time.time() - (len(sample_buf) / self.sample_rate)
                seg = {"clear_frame": clean, "timestamp": ts}
                try:
                    self.segment_q.put_nowait(seg)
                except queue.Full:
                    pass

        try:
            self.segment_q.put_nowait(None)
        except Exception:
            pass

    def asr_consumer_thread(self):
        SPEECH_HANG_MS = 300
        last_speech_time = None

        while True:
            item = self.segment_q.get()
            if item is None:
                break
            clear_frame = item["clear_frame"]
            ts = item["timestamp"]
            pcm = (np.clip(clear_frame, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(pcm, 16000)

            # считаем энергию
            energy = np.sqrt(np.mean(clear_frame**2))  # RMS
            THRESHOLD = 0.15
            has_audio = energy > THRESHOLD

            if is_speech:
                last_speech_time = ts

                accepted = self.rec.AcceptWaveform(pcm)
                if accepted:
                    out = json.loads(self.rec.Result())
                    text = out.get("text", "")
                    if text:
                        print(f"[RESULT] {text}")
                        # print(f"[TRANSLATION] {self.translator.translate(text)}", end="\r")
                else:
                    pres = json.loads(self.rec.PartialResult())
                    partial = pres.get("partial", "")
                    if partial:
                        print(f"[PARTIAL] {partial}", end="\r")
                        # print(f"[TRANSLATION] {self.translator.translate(partial)}", end="\r")
            else:
                # Если PCM состоит из нулей, проверяем нужно ли финализировать результат
                if last_speech_time is not None:
                    if time.time() - last_speech_time > (SPEECH_HANG_MS / 1000.0):
                        out = json.loads(self.rec.FinalResult())
                        text = out.get("text", "")
                        if text:
                            latency = time.time() - last_speech_time
                            print(f"\n[SUBTITLE] {text}   (latency ~{latency:.3f}s)")
                        last_speech_time = None

    def start(self):
        SetLogLevel(0)
        print("Launching a prototype of real-time subtitling...")
        print("Hit Ctrl+C to stop\n")

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                                 callback=self.audio_callback):
                producer = threading.Thread(target=self.producer_thread, daemon=True)
                consumer = threading.Thread(target=self.asr_consumer_thread, daemon=True)

                producer.start()
                consumer.start()

                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nОстановка по запросу пользователя...")
            # self.neural_speech_enhancer.save_models()
           
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            # self.neural_speech_enhancer.save_models()
 
        print("Программа завершена.")

