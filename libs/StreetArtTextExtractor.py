import easyocr, cv2
import language_tool_python
from spellchecker import SpellChecker
import hashlib, os
from typing import Dict

class StreetArtTextExtractor:
    def __init__(self):
        """
        Initialize the text extractor with Easy OCR.
        """
        self.reader = easyocr.Reader(['en'])
        self.spell_tool = SpellChecker()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        
    def _preprocess_image(self, image: cv2.typing.MatLike, preprocess_type: str) -> cv2.Mat:
        """
        Preprocess the image for better OCR results based on the specified type.
        :param image: cv2 image object.
        :param preprocess_type: Type of preprocessing to apply seperated by ','. Options are:
                                - "none": No preprocessing (over-writes all options).
                                - "grayscale": Convert image to grayscale.
                                - "contrast": Adjust contrast.
                                - "histogram": Apply histogram equalization.
                                - "blur": Apply Gaussian blur.
                                - "binary_th": Apply binary thresholding.
                                - "adaptive_th": Apply adaptive thresholding.
                                - "morphology": Apply morphological transformations.
        :param max_width: Maximum width for resizing.
        :param max_height: Maximum height for resizing.
        :return: Preprocessed image.
        """

        processed_image = image.copy()

        for proc_type in [i.strip() for i in preprocess_type.split(",")]:
            if "grayscale" == proc_type:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            elif "contrast" == proc_type:
                processed_image = cv2.convertScaleAbs(processed_image, alpha=1.3, beta=0) 
            elif "histogram" == proc_type:
                processed_image = cv2.equalizeHist(processed_image)
            elif "blur" == proc_type:
                processed_image = cv2.GaussianBlur(processed_image, (3, 3), 1)
            elif "binary_th" == proc_type:
                _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif "adaptive_th" == proc_type:
                processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            elif "morphology" == proc_type:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel, iterations=1)        
        return processed_image
    
    def fix_spelling(self, text: str) -> str:
        """
        Check and correct spelling in the text using PySpellChecker.
        :param text: Text to be checked.
        :return: Corrected text.
        """
        words = text.split()
        corrected = [self.spell_tool.correction(word) if self.spell_tool.correction(word) else word for word in words]
        return ' '.join(corrected)

    def fix_grammar(self, text: str) -> str:
        """
        Check and correct grammar in the text using LanguageTool.
        :param text: Text to be checked.
        :return: Corrected text.
        """
        matches = self.grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    
    def process_batch_of_images(self, images: list, preprocessing: str = None, score_th: float = 0.1) -> Dict[int, str]:
        """
        Process a batch of cv2 image objects and extract text from each.
        :param images: List of cv2 image objects.
        :param preprocess_type: Type of preprocessing to apply seperated by ','. Options are:
                                - "none": No preprocessing (over-writes all options).
                                - "grayscale": Convert image to grayscale.
                                - "contrast": Adjust contrast.
                                - "histogram": Apply histogram equalization.
                                - "blur": Apply Gaussian blur.
                                - "binary_th": Apply binary thresholding.
                                - "adaptive_th": Apply adaptive thresholding.
                                - "morphology": Apply morphological transformations.
        :return: Dictionary with indices as keys and extracted text as values.
        """
        
        results = {}
        for idx, image in enumerate(images):
            if isinstance(image, (cv2.typing.MatLike, type(None))):  # Ensure it's a valid cv2 image object
                if image is not None:
                    imghash = hashlib.sha1(str(image).encode(), usedforsecurity=False).hexdigest()
                    if preprocessing:
                        image = self._preprocess_image(image, preprocessing)
                    
                    # Use EasyOCR to extract text
                    result = self.reader.readtext(image, decoder='wordbeamsearch', contrast_ths=0, adjust_contrast=0.8)
                    
                    # Join the results into a single string
                    original_text = ' '.join([text[1] if text[2] > score_th else '--' for text in result])
                    results[imghash] = original_text

                    # corrected_text = self.fix_grammar(self.fix_spelling(original_text))
                    # results[imghash] = {
                    #     "original": original_text,
                    #     "corrected": corrected_text,
                    # }
        return results
                    

    def test_extract_text_with_easyocr(self, folder_path: str) -> Dict[str, str]:
        """ $$ Has results but questionable $$
        Extract text from all images in a folder using EasyOCR.
        :param folder_path: Path to the folder containing images.
        :return: Dictionary with hash of filenames as keys and extracted text as values.
        """
        results = {}
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image = cv2.imread(file_path) # Read image
                if image is not None:
                    result = self.reader.readtext(image,detail=0)
                    # Join the results into a single string
                    results[filename] = ' '.join(result)
                    # hashlib.sha1(filename.encode(), usedforsecurity=False).hexdigest()
        return results
    
# def test_extract_text_with_tesseract(self, folder_path: str) -> Dict[str, str]:
#     """ $$ Probably not a usable model $$
#     Extract text from all images in a folder using Tesseract OCR.
#     :param folder_path: Path to the folder containing images.
#     :return: Dictionary with filenames as keys and extracted text as values.
#     """
#     results = {}
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             image = cv2.imread(file_path)
#             if image is not None:
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 blur = cv2.GaussianBlur(gray, (3,3), 0)
#                 thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#                 # Morph open to remove noise and invert image
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#                 opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#                 invert = 255 - opening
#                 text = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')  # Extract text using Tesseract
#                 results[filename] = text
#                 # hashlib.sha1(filename.encode(), usedforsecurity=False).hexdigest()
#     return results

# Example usage:
# extractor = StreetArtTextExtractor(tesseract_cmd="/usr/bin/tesseract")
# results = extractor.extract_text_from_images_in_folder("path_to_images_folder")
# print(results)