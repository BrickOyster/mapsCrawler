import hashlib, os, sys
from PIL import Image

# General tools
import torch, cv2
import numpy as np
from torch.autograd import Variable

# Craft bounding boxes detector
from .Craft.craft import CRAFT
from .Craft import craft_utils as craft_utils, imgproc as imgproc

# Pytorch crnn
from .Crnn import utils as crnn_utils, dataset as crnn_dataset
from .Crnn.models import crnn as crnn_models

from typing import Dict
from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

DBG_MODE = 0

class StreetArtTextExtractor:
    def __init__(self, detect: list[str] = ['en'], alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"):
        """
        Class to extract text from images using CRAFT and Pytorch.
        """
        # Craft bounding box detector
        self.craft = CRAFT()
        self.craft.load_state_dict(copyStateDict(torch.load(os.path.join(os.path.dirname(__file__), 'data/craft_mlt_25k.pth'), map_location='cpu', weights_only=False)))
        self.craft.eval()

        # Pytorch crnn text recognizer
        self.crnn = crnn_models.CRNN(32, 1, 37, 256)
        self.crnn.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'data/crnn.pth', ), map_location='cpu', weights_only=False))
        self.crnn.eval()
        self.crnn_converter = crnn_utils.strLabelConverter(alphabet)
        self.crnn_transformer = crnn_dataset.resizeNormalize((100, 32))
        

    def _preprocess_image(self, image: cv2.typing.MatLike, preprocess_type: str = "") -> cv2.Mat:
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
            else:
                pass     
        return processed_image
    
    def craft_detect(self, image: cv2.typing.MatLike, text_threshold: float = 0.7, link_threshold: float = 0.4, low_text: float = 0.4) -> tuple:
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        
        # forward pass
        with torch.no_grad():
            y, feature = self.craft(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text
    
    def process_batch_of_images(self, images: list, preprocessing: str = None, score_th: float = 0.07) -> Dict[int, str]:
        """
        Process a batch of cv2 image objects and extract text from each.
        :param images: List of cv2 image objects.
        :param preprocess_type: Type of preprocessing to apply seperated by ','. Options are:
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
                    bboxes, polys, score_text = self.craft_detect(image, 0.7, 0.4, 0.4)
                    
                    imghash = hashlib.sha1(f"{str(image)}{idx}{str(bboxes)}".encode(), usedforsecurity=True).hexdigest()
                    
                    for bbox in bboxes:
                        # Extract the region of interest (ROI) using the bounding box
                        x_min, y_min = np.min(bbox, axis=0).astype(int)
                        x_max, y_max = np.max(bbox, axis=0).astype(int)
                        height, width = image.shape[:2]
                        x_min = max(0, min(x_min, width - 1))
                        x_max = max(0, min(x_max, width - 1))
                        y_min = max(0, min(y_min, height - 1))
                        y_max = max(0, min(y_max, height - 1))
                        cropped = image[y_min:y_max, x_min:x_max]

                        # Skip empty or invalid bounding boxes
                        if cropped.size == 0:
                            continue

                        # Apply preprocessing if specified
                        if preprocessing:
                            cropped = self._preprocess_image(cropped, preprocessing)

                        # Ensure the ROI is grayscale
                        if len(cropped.shape) > 2:
                            roi = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        else:
                            roi = cropped

                        # Prepare the ROI for CRNN
                        roi = Image.fromarray(roi)
                        roi = self.crnn_transformer(roi)
                        roi = Variable(roi.unsqueeze(0))

                        # Perform text recognition using CRNN
                        with torch.no_grad():  # Disable gradient computation for inference
                            preds = self.crnn(roi)
                            _, preds = preds.max(2)
                            preds = preds.transpose(1, 0).contiguous().view(-1)

                        preds_size = Variable(torch.IntTensor([preds.size(0)]))
                        raw_pred = self.crnn_converter.decode(preds.data, preds_size.data, raw=True)
                        sim_pred = self.crnn_converter.decode(preds.data, preds_size.data, raw=False)

                        # Store the recognized text in the results dictionary
                        results[imghash] = results.get(imghash, "") + f" {sim_pred.strip()}"
                        print(f"\r Text extracted: {sim_pred.strip()[:30]}", end="")
                    #         results[imghash] = results.get(imghash, "") + f"|{sim_pred} / {''.join(result)}|"
                    #         # Test code
                    #         # Draw bounding boxes and recognized text on the image
                    #         box = np.array(bbox).astype(int)
                    #         text_position = (box[0][0], box[0][1] - 10 if box[0][1] - 10 > 10 else box[0][1] + 10)
                    #         cv2.putText(image, sim_pred, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

                    #     # Display the image with bounding boxes and text
                    #     box = np.array(bbox).astype(int)
                    #     cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

                    #     # Resize the image to fit the screen
                    #     screen_width = 1280
                    #     screen_height = 720
                    #     height, width = image.shape[:2]
                    #     scale = min(screen_width / width, screen_height / height)
                    #     resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

                    # cv2.imshow(f"Image {imghash}", resized_image)
                    # Allow interaction with the displayed image window
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
    print([file for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    results = extractor.process_batch_of_images(images)
    for imghash, entry in results.items():
        print(f"Image {imghash}:\n\n\t{entry}\n")
    
    # cv2.imshow('dummy',images[0])
    # while 1:
    #     if cv2.waitKey(0) & 0xFF == ord('p'):
    #         break
    # cv2.destroyAllWindows()
    sys.exit(0)