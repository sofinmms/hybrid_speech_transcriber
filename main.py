from real_time_transcriber import RealTimeTranscriber

def main():
    transcriber = RealTimeTranscriber(model_path="./vosk-model-small-ru-0.22")
    print("Ready to record and translate speech...")
    try:
        transcriber.start()
    except KeyboardInterrupt:
        print("\Completing the program...")

if __name__ == "__main__":
    main()
