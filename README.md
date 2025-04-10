# StreetArtTextExtractor

StreetArtTextExtractor is a Python library designed to extract and process text from images and categorize them based on text content, particularly for street art and similar use cases. It leverages EasyOCR for text recognition, OpenCV for image preprocessing, and additional tools for spelling correction.

## Features

- **Text Extraction**: Extract text from images using EasyOCR.
- **Image Preprocessing**: Enhance images for better OCR results with options like grayscale conversion, contrast adjustment, histogram equalization, Gaussian blur, and more.
- **Batch Processing**: Process multiple images at once and extract text from each.
- **Spelling Correction**: Automatically correct spelling errors in extracted text using PySpellChecker.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brickoyster/mapsCrawler.git
cd mapsCrawler
```

2. Ensure you have the following installed:
    - OpenCV (cv2)
    - EasyOCR
    - PySpellChecker

    Can be done with
```bash
pip install -r requirements.txt
```

## Usage

Example code:

```python
import libs.StreetArtTextExtractor as StreetArtTextExtractor
import cv2
import os

# Initialize the extractor
extractor = StreetArtTextExtractor.StreetArtTextExtractor()

# Load images from a folder (ie img/)
image_folder = "img/"
images = [cv2.imread(os.path.join(image_folder, file)) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Process images and extract text
results = extractor.process_batch_of_images(images, preprocessing="contrast")

# Print results
for imghash, entry in results.items():
    print(f"Image {imghash}:\n\n\t{entry}\n")
```

## Preprocessing Options

You can specify multiple preprocessing steps separated by commas. Available options:
- **none**: No preprocessing.
- **grayscale**: Convert the image to grayscale.
- **contrast**: Adjust the image contrast.
- **histogram**: Apply histogram equalization.
- **blur**: Apply Gaussian blur.
- **binary_th**: Apply binary thresholding.
- **adaptive_th**: Apply adaptive thresholding.
- **morphology**: Apply morphological transformations.

## Grammar and Spelling Correction

The library includes methods to correct spelling and grammar in extracted text:
- `fix_spelling(text: str) -> str`: Corrects spelling errors.
- `fix_grammar(text: str) -> str`: Corrects grammar issues.

## Batch Processing

The `process_batch_of_images` method allows you to process multiple images at once. It returns a dictionary where the keys are image hashes and the values are the extracted text.
