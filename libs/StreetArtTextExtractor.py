import hashlib, os
import easyocr, cv2
from typing import Dict

class StreetArtTextExtractor:
    def __init__(self, detect: list[str] = ['en']):
        """
        Class to extract text from images using Easy OCR.
        PySpellChecker for spelling correction.
        LanguageTool for grammar correction.
        """
        self.reader = easyocr.Reader(detect)
        
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
    
    def process_batch_of_images(self, images: list, preprocessing: str = None, score_th: float = 0.07) -> Dict[int, str]:
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
                    result = self.reader.readtext(image, decoder='wordbeamsearch')
                    
                    # Join the results into a single string
                    original_text = ' '.join([text[1] if text[2] > score_th else '--' for text in result])
                    results[imghash] = original_text
                    
        return results
    
# Example usage
if __name__ == "__main__":
    extractor = StreetArtTextExtractor()

    # Read all images in a folder and extract text
    # folder_path = "img/" # Path to the folder containing images
    # results = extractor.test_extract_text_with_easyocr(folder_path)
    # for filename, text in results.items():
    #     print(f"Image {filename}: {text}")
    
    # Or make a list of images 
    image_folder = "img/"
    images = [cv2.imread(os.path.join(image_folder, file)) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    results = extractor.process_batch_of_images(images, preprocessing="contrast")
    for imghash, entry in results.items():
        print(f"Image {imghash}:\n\n\t{entry}\n")