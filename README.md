# StreetArtTextExtractor

StreetArtTextExtractor is a Python library designed to extract and process text from images and categorize them based on text content, particularly for street art and similar use cases. It leverages CRAFT for text detection, PyTorch CRNN for text recognition, OpenCV for image preprocessing, and additional tools for spelling correction.

## Features

- **Text Extraction**: Extract text from images using CRAFT and PyTorch CRNN.
- **Image Preprocessing**: Enhance images for better OCR results with options like grayscale conversion, contrast adjustment, histogram equalization, Gaussian blur, and more.
- **Batch Processing**: Process multiple images at once and extract text from each.
- **Spelling Correction**: Automatically correct spelling errors in extracted text using PySpellChecker.
- **Text processing**: ...

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brickoyster/mapsCrawler.git
cd mapsCrawler
```

2. Install requirements
```bash
pip install -r requirements.txt
pip install mapillary --no-deps
```

## Usage

Example code:

```python
# To write later
```

## Image Preprocessing Options

You can specify multiple preprocessing steps separated by commas. Available options:
- **none**: No preprocessing.
- **grayscale**: Convert the image to grayscale.
- **contrast**: Adjust the image contrast.
- **histogram**: Apply histogram equalization.
- **blur**: Apply Gaussian blur.
- **binary_th**: Apply binary thresholding.
- **adaptive_th**: Apply adaptive thresholding.
- **morphology**: Apply morphological transformations.

## Text processing

- `fix_spelling(text: str) -> str`: Corrects spelling errors.
- `translate_text(input_text: str) -> str`: Translates text to the target language.
- `do_profanity_check(input_text: dict) -> tuple`: Detects profanity in the extracted text and returns the count of profane and cleared texts.

## Batch Processing

The `process_batch_of_images` method allows you to process multiple images at once. It returns a dictionary where the keys are image hashes and the values are the extracted text.

## Requirements

- Python 3.8+
- OpenCV
- CRAFT dependencies
- Pytorch Crnn dependencies
- PySpellChecker
- Deep Translator
- Joblib (for profanity detection models)

## Acknowledgments

- [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
- [Pytorch crnn](https://github.com/meijieru/crnn.pytorch)
- [OpenCV](https://opencv.org/)
- [PySpellChecker](https://github.com/barrust/pyspellchecker)
- [Deep Translator](https://github.com/nidhaloff/deep-translator)
- [Joblib](https://joblib.readthedocs.io/)