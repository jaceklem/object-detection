import os
import numpy as np

from detector import Detector
from multiObjectTracking import MultiObjectTracking

from detector_RCNN import Detector as RCNN

from utilsMOT import processDetections

import platform
import pathlib

# Check the operating system and set the appropriate path type
# needed when YOLO weights are trained on a different OS
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Suppress FutureWarning from YOLO: 'autocast'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

def main(inputPath, weightPath, annotationsPath, imageFactor, bubbleGrowth, toggleDouwe, debugMode):
    # initialization
    stateDim = 4 # [xpos, ypos, xvel, yvel]
    measDim = 2 # [xmeasurement, ymeasurement]

    if bubbleGrowth:
        stateDim = 5 # [xpos, ypos, xvel, yvel, size]
        measDim = 3 # [xmeasurement, ymeasurement, size]

    # inputs
    processNoiseCov = np.eye(stateDim) * 0.1
    measurementNoiseCov = np.eye(measDim) * 0.1
    confThreshold = 0.9
    iouThresh = 0.5
    maxDist = 5 * (1 - confThreshold)
    dt = 1

    # initialize the managers
    if toggleDouwe:
        detector = RCNN('Code/EKF/Model/model.pth')
    
    else:    
        detector = Detector(weightPath, confThreshold, iouThresh, debugMode)
    
    trackerManager = MultiObjectTracking(stateDim, measDim, processNoiseCov, measurementNoiseCov, confThreshold, bubbleGrowth, iouThresh, maxDist, debugMode)

    if os.path.isdir(inputPath):
        # check and collect images
        imageFiles = [os.path.join(inputPath, f) for f in os.listdir(inputPath)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        imageFiles.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))  
        if not imageFiles:
            print(f"No images found in the directory {inputPath}")
            return
        
        print(f"Found {len(imageFiles)} image(s) in {inputPath}")
        processDetections(imageFiles, detector, trackerManager, annotationsPath, imageFactor, dt, toggleDouwe, debugMode)
        
    elif os.path.isfile(inputPath):
        processDetections([inputPath], detector, trackerManager, annotationsPath, imageFactor, dt, toggleDouwe, debugMode)

    else:
        print(f"Error: {inputPath} is not a valid file or directory.")

if __name__ == "__main__":
    """
    Input can be path to image directory or to a video file.

    Notes:
    Toggle debug mode in most files.
    Change parameters in function main.
    When using generated images, change plot attributes to pixels instead of mm, and vice versa
    """
    # Useful for dataset not in the same folder
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # input files
    video = 24
    inputFiles = f'Code/EKF/datasets/trackingNew/Images/video_{video}'
    annotationsPath = f'Code/EKF/datasets/trackingNew/Annotations/video_{video}'

    # Josefine's dataset
    # inputFiles = 'Code/EKF/datasets/Josefine/22_02_22'
    # annotationsPath = '' # no annotations provided

    # YOLO weights
    experiment = "exp1"
    weightPath = os.path.join(root, "Code", "EKF", "yolov5", "runs", "train", experiment, "weights", "best.pt")

    # Image scaling factor 
    # For Josefine's dataset use factor = 17
    factor = 1
    
    # Toggle bubble growth
    toggleBubbleGrowth = False

    # Toggle Douwe's PyTorch model
    toggleDouwe = False

    # debugging
    debugMode = False

    # call the function
    main(inputFiles, weightPath, annotationsPath, factor, toggleBubbleGrowth, toggleDouwe, debugMode)