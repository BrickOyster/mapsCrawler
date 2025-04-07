import easyocr
import cv2
import mapillary.interface as mly
import json
from typing import Dict
import os

class StreetArtTextExtractor:
    def __init__(self):
        """
        Initialize the text extractor with Easy OCR.
        """
        self.reader = easyocr.Reader(['en'])
        
    def _preprocess_image(self, image: cv2.typing.MatLike, preprocess_type: str) -> cv2.Mat:
        """
        Preprocess the image for better OCR results based on the specified type.
        :param image: cv2 image object.
        :param preprocess_type: Type of preprocessing to apply seperated by ','. Options are:
                                - "none": No preprocessing (over-writes all options).
                                - "grayscale": Convert image to grayscale.
                                - "blur": Apply Gaussian blur.
                                - "threshold": Apply binary thresholding.
                                - "morphology": Apply morphological transformations.
        :return: Preprocessed image.
        """
        if not preprocess_type:
            return image
        
        preprocess_types = [i.strip() for i in preprocess_type.split(",")]
        if "grayscale" in preprocess_types:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if "blur" in preprocess_types:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if "threshold" in preprocess_types:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if "morphology" in preprocess_types:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return image
    
    def process_batch_of_images(self, images: list, preprocessing: str = None) -> Dict[int, str]:
        """
        Process a batch of cv2 image objects and extract text from each.
        :param images: List of cv2 image objects.
        :param preprocess_type: Type of preprocessing to apply seperated by ','. Options are:
                                - "none": No preprocessing (over-writes all options).
                                - "grayscale": Convert image to grayscale.
                                - "blur": Apply Gaussian blur.
                                - "threshold": Apply binary thresholding.
                                - "morphology": Apply morphological transformations.
        :return: Dictionary with indices as keys and extracted text as values.
        """
        
        results = {}
        for idx, image in enumerate(images):
            if isinstance(image, (cv2.typing.MatLike, type(None))):  # Ensure it's a valid cv2 image object
                image = self._preprocess_image(image, preprocessing)
                if image is not None:
                    # Use EasyOCR to extract text
                    result = self.reader.readtext(image, detail=0)
                    # Join the results into a single string
                    results[idx] = ' '.join(result)
                    # hashlib.sha1(filename.encode(), usedforsecurity=False).hexdigest()
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