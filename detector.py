import torch
import torchvision as tv
import cv2
import numpy as np

from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image

class Detector():
    """
    A class to perform object detection on video frames.

    Attributes:
        weightPath (str): Path to the model weights file.
        confThresh (float): Confidence threshold for detection.
        iouThresh (float): Intersection over Union (IoU) threshold for detection.
        debugMode (bool): Flag to enable or disable debug mode.
        douwe (bool): Flag to determine which model to use.
        device (str): Device to run the model on ('cuda' for GPU, 'cpu' for CPU).
        model (torch.nn.Module): The loaded detection model.
    """
    def __init__(self, weightPath, confThresh, iouThresh, debugMode):
        """
        Initialize the detector with the given parameters.
        
        Args:
            weightPath (str): Path to the model weights file.
            confThresh (float): Confidence threshold for detection.
            iouThresh (float): Intersection over Union (IoU) threshold for detection.
            douwe (bool): Flag to determine which model to use. If True, use a custom model; otherwise, use YOLOv5.
            debugMode (bool): Flag to enable or disable debug mode.
        """
        self.weightPath = weightPath
        self.confThresh = confThresh
        self.iouThresh = iouThresh
        self.debugMode = debugMode

        # check which device to use for detection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # let user know which device
        if self.device == 'cpu':
            print("Warning: Running on CPU might be slower...")

        else:
            print("Runninig on GPU...")

        # load yolov5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weightPath, force_reload=True, trust_repo=True)

        # switch to correct device
        self.model.to(self.device)

    def detect(self, frame):
        """
        Detects objects in the given frame using YOLO.

        Args:
            frame (ndarray): Input image frame.
            debugMode (bool): toggle debug mode

        Returns:
            centers (np.array): Numpy array of detected centers.
        """
        try:
            # Frame formatting for YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()

            if self.debugMode:
                print(f"Detection Results: {results.pandas().xywh[0]}")
                print(f"Number of detections before NMS: {len(detections)}")

            # convert to NMS format
            boxes = detections[:, :4] # [xmin, ymin, xmax, ymax]
            scores = detections[:, 4]
            classId = detections[:, 5]

            # convert to tensor
            boxes = torch.tensor(boxes).float().to(self.device)
            scores = torch.tensor(scores).float().to(self.device)

            # apply NMS
            indices = nms(boxes, scores, self.iouThresh)

            # filter detections
            filteredDetections = []
            for idx in indices.cpu().numpy():
                x1, y1, x2, y2 = detections[idx, :4]
                conf = detections[idx, 4]
                filteredDetections.append([x1, y1, x2, y2, conf])

            return filteredDetections

        except Exception as e:
            print(f"Error during detection: {e}")
            return []