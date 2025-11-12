from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, source_lang='auto', target_lang='en'):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text):
        try:
            translated = GoogleTranslator(source=self.source_lang, target=self.target_lang).translate(text)
            return translated
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return None
