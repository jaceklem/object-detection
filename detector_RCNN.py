import torch
import numpy as np
from PIL import Image

class Detector:
    def __init__(self, model, confThresh=0.9):
        """
        Initializes the detector_RCNN class.

        Args:
            model (str): Path to the model file to be loaded.
            confThresh (float, optional): Confidence threshold for detections. Defaults to 0.9.

        Attributes:
            model (torch.nn.Module): The loaded model.
            confThresh (float): The confidence threshold for detections.
        """
        self.model = torch.load(model, map_location=torch.device('cpu'))
        self.confThresh = confThresh

    def convert_image(self, img_path):
        """
        Convert an image to a tensor suitable for model input.
        This function loads an image from the specified file path, converts it to RGB format,
        normalizes the pixel values to the range [0, 1], and converts it to a PyTorch tensor
        with the shape (C, H, W).
        
        Args:
            img_path (str): The file path to the image.
        
        Returns:
            torch.Tensor: The image as a tensor with shape (C, H, W) and dtype float32.
        """
        # Load the image
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.asarray(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
            
        return img

    def array_to_image(self, array):
        """
        Converts a NumPy array to a PyTorch tensor representing an image.

        Args:
            array (np.ndarray): Input image as a NumPy array.

        Returns:
            torch.Tensor: Image as a PyTorch tensor with shape (C, H, W) and values normalized to [0, 1].
        """
        # Convert image to PIL image
        img = Image.fromarray(array.astype(np.uint8))
        img = img.convert("RGB")
        img = np.asarray(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        return img

    def detect(self, img):
        """
        Detect objects in the given image using the pre-trained model.

        Args:
            img (PIL.Image or Tensor): The input image for object detection.

        Returns:
            numpy.ndarray: An array of bounding boxes with high confidence. Each bounding box is represented as 
                           [xmin, ymin, xmax, ymax].
        """
        # prediction
        pred = self.model([img, ])

        # bounding boxes are in the format [xmin, ymin, xmax, ymax]
        boxes = pred[0]["boxes"].detach().numpy()
        scores = pred[0]["scores"].detach().numpy()

        # remove boxes with low confidence
        boxes = boxes[scores > self.confidence]

        return boxes