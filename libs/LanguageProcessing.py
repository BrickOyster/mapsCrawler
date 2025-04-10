from  deep_translator import GoogleTranslator
import numpy as np
from my_profanity_check import predict, predict_prob
from spellchecker import SpellChecker

class LanguageProcessing:
    def __init__(self, lang = ('english', 'en')):
        """
        Class to process the text extracted from the images.
        Contains a module for language translation  to english and modules for vocabulary and grammar fixes
        post the OCR processing
        :param
        """
        self.language, self.language_code = lang
        self.spell_tool = SpellChecker(language=self.language_code)
        self.translator = GoogleTranslator(source='auto', target=self.language_code)
        
    def fix_spelling(self, text: str) -> str:
        """
        Check and correct spelling in the text using PySpellChecker.
        :param text: Text to be checked.
        :return: Corrected text.
        """
        words = text.split()
        corrected = [self.spell_tool.correction(word) if self.spell_tool.correction(word) else word for word in words]
        return ' '.join(corrected)
        
    def translate_text(self, input_text):
        """
        Function to translate input text
        :param input_text: text to be translated
        :return: translated text
        """
        try:
            translated = self.translator.translate(input_text)
            return translated
        except Exception as e:
            print(f"Error during translation: {e}")
            return input_text
        
    def do_profanity_check(self, input_text):
        """
        Function to do a profanity check
        :param input_text: text to be checked
        :return: profanity counter and cleared images counter
        """
        profanity_counter = 0
        cleared_images_text = []
        clear_images_counter = 0

        for _,text in input_text.items():
            if predict(text)[0] == 1:
                profanity_counter += 1
            else:
                cleared_images_text.append(text)

        for clear_text in cleared_images_text:
            translation = self.translate_text(clear_text)
            if predict(translation)[0] == 1:
                profanity_counter += 1
            else:
                clear_images_counter += 1

        return profanity_counter, clear_images_counter
    
# dummy main function to test the class
if __name__ == "__main__":
    lp = LanguageProcessing()
    text = "Bonjour tout le monde"
    translated_text = lp.translate_text(text, source_language='fr')
    print(f"Translated text: {translated_text}")
    print(lp.fix_spelling(translated_text))
    # pc, cic = lp.do_profanity_check({"image1": "This is a test", "image2": "This is  fuck test"})
    # print(f"{pc} ,{cic}")
