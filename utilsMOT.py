import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import norm
import matplotlib.cm as cm

import imageio.v3 as iio
import shutil

import xml.etree.ElementTree as ET

from torchvision.ops import box_iou

def processDetections(imageFiles, detector, trackerManager, annotationsPath, imageFactor, dt, toggleDouwe, debugMode):
    """
    Processes a list of images and saves the output as a video.
    
    Args:
        imageFiles (list of str): List of image file paths.
        detector (object): Object detector instance.
        trackerManager (object): Tracker manager instance.
        dt (int): Time steps between images.
        debugMode (bool): Toggle debug mode

    Returns:
        None
    """
    try:
        if not imageFiles:
            print("No images provided.")
            return

        # process time
        start = time.time()
        print('Start processing...')

        # Read the first valid image to determine frame size
        firstFrame = None
        frameWidth, frameHeight = None, None

        for imageFile in imageFiles:
            frame = cv2.imread(imageFile)

            if frame is None:
                print(f"Error: Unable to load image {imageFile}")

            frameHeight, frameWidth = frame.shape[:2]
            firstFrame = frame
            break

        if firstFrame is None:
            print(f"Error: Unable to find valid images.")
            return

        # root of object-detection
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resultsPath = os.path.join(root, "Code", "EKF", "Results")

        # initialize output dirs and clean them
        outputPathSD = os.path.join(resultsPath, "sizeDist")
        outputPathSizes = os.path.join(resultsPath, "Sizes")
        outputPathBC = os.path.join(resultsPath, "bubbleCount")
        outputPathAvgDiams = os.path.join(resultsPath, "avgBubbleDiams")
        outputPathIouValues = os.path.join(resultsPath, "iouValues")
        outputFrames = os.path.join(resultsPath, "outputFrames")

        dirs = [outputPathSD, outputPathSizes, outputPathBC, outputPathAvgDiams, outputPathIouValues, outputFrames]
        
        # clean the directories
        cleanDirectory(dirs)
        os.makedirs(outputFrames, exist_ok=True)

        # initialize
        bubbleSizes = []
        bubbleCounts = []
        confidences = []
        avgDiams = []

        for idx, imageFile in enumerate(imageFiles):
            frame = cv2.imread(imageFile)

            if frame is None:
                print(f"Warning: Unable to load image {imageFile}. Skipping.")
                continue

            # ensure correct sizing of frames
            if frame.shape[0] != frameHeight or frame.shape[1] != frameWidth:
                print(f"Resizing frame to ({frameWidth}, {frameHeight})...")
                frame = cv2.resize(frame, (frameWidth, frameHeight), interpolation=cv2.INTER_AREA)

            # process frames
            trackerManager, centers, frameDimensions = processFrame(frame, detector, trackerManager, bubbleSizes, confidences, idx, toggleDouwe, debugMode)
            filteredCenters = removeDoubleDet(centers)
            sizes, diameters, _ = detBubbleSizes(filteredCenters, imageFactor)
            drawPositions(frame, filteredCenters)

            # calculate IoU values
            if len(annotationsPath) > 0:
                annotations = loadAnnotations(annotationsPath, idx)
                iouValues = calculateIOU(filteredCenters, annotations, idx, outputPathIouValues)
            
            # get mean sizes
            avgDiam = np.mean(diameters)

            # store data
            bubbleCounts.append(len(filteredCenters))
            avgDiams.append(avgDiam)

            # plot bubble sizes per frame
            sizeDist(diameters, idx, outputPathSD)
            plotBubbleSizes(sizes, idx, outputPathSizes, dt)

            # saves the frames
            frameFileName = os.path.join(outputFrames, f"frame_{idx:04d}.png")
            cv2.imwrite(frameFileName, frame)

            # Log progress every 10 frames
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(imageFiles)} images...")

        # final print statement
        print(f"Processed {len(imageFiles)}/{len(imageFiles)} images and saved the frames.")

        # plotting
        print("Generating plots and making GIF...")
        plotBubbleCount(bubbleCounts, outputPathBC, dt)
        plotAvgBubbleDia(avgDiams, outputPathAvgDiams, dt)
        makeGif(outputFrames)
        end = time.time()
        print(f"That's it. Total run time {(end - start):.5f} seconds.")

    except Exception as e:
        print(f"Error: {e}")

def processFrame(frame, detector, trackerManager, bubbleSizes, confidences, idx, toggleDouwe, debugMode):
    """
    Processes a single frame by running object detection and updating the tracker.
    
    Args:
        frame (numpy.ndarray): The input frame to process.
        detector (object): The object detector with a `detect` method that takes a frame and returns detected centers.
        trackerManager (object): The tracker manager with an `update` method that takes detected centers.
        bubbleSizes (list): List of bubble sizes for detected objects.
        confidences (list): List of confidence scores for detected objects.
    
    Returns:
        tuple: A tuple containing the updated trackerManager and the detected centers.
    """
    try:
        # get initial sizes
        height, width = frame.shape[:2]
        allDetections = []

        # quadrant processing for large frames
        if width > 1024 or height > 1024:
            print("Starting quadrant processing...")

            combs = [
                (0, 0, width // 2, height // 2),        # Top-left
                (width // 2, 0, width, height // 2),    # Top-right
                (0, height // 2, width // 2, height),   # Bottom-left
                (width // 2, height // 2, width, height)  # Bottom-right
            ]

            for (x1, y1, x2, y2) in combs:
                quadrant = frame[y1:y2, x1:x2]
                quadrantDetections = detector.model(quadrant).xyxy[0].cpu().numpy()

                if quadrantDetections is not None and quadrantDetections.size > 0:
                    # adjust coordinates
                    quadrantDetections[:, [0, 2]] += x1 
                    quadrantDetections[:, [1, 3]] += y1
                    allDetections.extend(quadrantDetections)

            detections = allDetections

        elif toggleDouwe:
            img = detector.array_to_image(frame)
            pred = detector.model([img, ])

            # bounding boxes are in the format [xmin, ymin, xmax, ymax]
            boxes = pred[0]["boxes"].detach().numpy()
            scores = pred[0]["scores"].detach().numpy()

            # remove boxes with low confidence
            detections = boxes[scores > detector.confThresh] 
            scores = scores[scores > detector.confThresh] 

            # add the scores at the end
            detections = np.insert(detections, 4, scores, axis=1)
            
        else:
            detections = detector.model(frame).xyxy[0].cpu().numpy()
        
        # validation
        if detections is None or len(detections) == 0:
            print("No detections found.")
            return trackerManager, [], bubbleSizes, confidences, {}, (width, height)

        # Update tracker with detected centers
        centers = []
        for det in detections:
            # validation
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]

                # validation
                if conf < detector.confThresh:
                    continue

                centers.append([x1, y1, x2, y2, conf])

            else:
                print(f"Invalid detection format: {det}")

        # get more information
        if debugMode:
            print(f"Detected centers: {centers}")
            print(f"Process frame shape: {frame.shape}")

        # update the trackers        
        trackerManager.update(centers, idx)

        return trackerManager, centers, (width, height)
    
    except Exception as e:
        print(f"Error processing frame: {e}")

        return trackerManager, [], (0, 0)

def drawPositions(frame, centers):
    """
    Draws the estimated and measured positions on the frame for multiple trackers.

    Args:
        frame (ndarray): The video frame to draw on.
        centers (list): List of detected centers for each object.

    Returns:
        None
    """
    # for better line visibility
    if np.shape(frame)[1] > 800:
        thickness = 10

    else:
        thickness = 2

    # chance color of box
    color = [5, 130, 30] # (b, g, r)

    # draw detected centers (measured positions)
    for center in centers:
        x1, y1, x2, y2 = center[:4]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)  # detected position

def removeDoubleDet(detections):
    """
    Remove duplicate detections based on the distance between their centers.
   
    Args:
        detections (list): A list of detections containing x and y coordinates

    Returns:
        list: A list of unique detections with duplicates removed.
    """
    uniqueDetections = []

    for det1 in detections:
        x1, y1, x2, y2 = det1[:4]

        # get centers
        x1Center = (x1 + x2) / 2
        y1Center = (y1 + y2) / 2

        isDuplicate = False

        for det2 in uniqueDetections:
            x3, y3, x4, y4 = det2[:4]
            
            # get centers
            x2Center = (x3 + x4) / 2
            y2Center = (y3 + y4) / 2

            distance = np.sqrt((x1Center - x2Center) ** 2 + (y1Center - y2Center) ** 2)

            if distance < 10:
                isDuplicate = True
                break

        if not isDuplicate:
            uniqueDetections.append(det1)

    return uniqueDetections

def detBubbleSizes(detections, imageFactor, debugMode=False):
    """
    Calculate the bubble sizes for a list of detections.
    Each detection is expected to be a tuple containing:
    (xmin, ymin, xmax, ymax, conf, classLabel)
    
    Args:
        detections (list of tuples): A list where each element is a tuple representing a detection.
        debugMode (bool): Toggle debug mode.
    
    Returns:
        sizes (list): A list containing the sizes of the bubbles
        confidences (list): A list containing confidence scores
    """
    # initialize
    bubbleDict = {}
    sizes = []
    diameters = []
    confidences = []

    # gather data
    for i, det in enumerate(detections):
        if len(det) < 4:
            print(f"Skipping invalid detections: {det}")
            continue

        # extract data
        xmin, ymin, xmax, ymax = det[:4]
        conf = det[4]

        # calculations
        width = (xmax - xmin) / imageFactor
        height = (ymax - ymin) / imageFactor
        diameter = (width + height) / 2 # a bit more precision
        radius = (width + height) / 4 # a bit more precision than just w/2
        area = np.pi * radius ** 2

        sizes.append(area)
        diameters.append(diameter)
        confidences.append(conf)

        # store data
        bubbleDict[i] = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'conf': conf,
            'size': area
        }

        if debugMode:
            print(f"Detection {i}: {bubbleDict[i]}\n")

    return sizes, diameters, confidences

def sizeDist(bubbleSizes, idx, outputPath):
    """
    Plots the distribution of bubble sizes and saves the results to a CSV file and a PNG image.
    
    Args:
        bubbleSizes (list or array-like): A list or array of bubble sizes to plot.
        idx (int): Iteration index
        outputPath (str): Output path where the file should be stored.

    Returns:
        None
    """
    if not bubbleSizes:
        print('No bubble sizes to plot.')
        return
    
    # convert to numpy array and sort
    bubbleSizes = np.array(bubbleSizes)
    bubbleSizes = np.sort(bubbleSizes)
    
    # plot results
    plt.figure(figsize=(12,8))
    points = np.linspace(bubbleSizes.min(), bubbleSizes.max(), 101)

    # probability density function
    pdf = norm.pdf(points, np.mean(bubbleSizes), np.std(bubbleSizes))
    plt.plot(points, pdf, color='r')

    # determine bin sizes
    if len(bubbleSizes) < 50:
        bins = max(7, int(np.sqrt(len(bubbleSizes))))

    elif len(bubbleSizes) <= 300:
        bins = np.ceil(np.log2(len(bubbleSizes) + 1)).astype(int) # Sturges's rule

    else:
        iqr = np.percentile(bubbleSizes, 75) - np.percentile(bubbleSizes, 25)
        binWidth = 2 * iqr * (len(bubbleSizes ** (-1/3))) # Freedman-Diaconis rule
        bins = int((max(bubbleSizes) - min(bubbleSizes)) / binWidth)
        bins = max(10, bins)

    # plotting
    plt.hist(bubbleSizes, bins=bins, alpha=0.75, edgecolor='black', density=True)
    plt.title(f'Bubble Diameter Distribution Frame: {idx:04d}')
    plt.xlabel('Bubble diameter [pixels]')
    plt.ylabel('Probability density')
    plt.grid(axis='y', linestyle='--')
    
    # save the results
    os.makedirs(outputPath, exist_ok=True)
    figurePath = os.path.join(outputPath, f'bubbleDiaDist_frame{idx:04d}.png')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()

    # save the sizes as csv file
    csvPath = os.path.join(outputPath, f'bubbleDia_frame{idx:04d}.csv')
    np.savetxt(csvPath, bubbleSizes, delimiter=',', header='Bubble Sizes (pixels²)')

def plotBubbleCount(bubbleCounts, outputPath, dt):
    """
    Plots the number of nucleation sites (bubble counts) over time and saves the plot as an image file.
    
    Args:
        bubbleCounts (list of int): A list containing the bubble counts at each time point.
        outputPath (str): Output path where the file should be stored.
        dt (float): The time interval between each bubble count measurement.
    
    Returns:
        None
    """
    # generate time points
    timePoints = [i * dt for i in range(len(bubbleCounts))]

    # plotting
    plt.figure(figsize=(12,8))
    plt.scatter(timePoints, bubbleCounts, color='red')
    plt.title("Number of nucleation sites")
    plt.xlabel("Time [min]")
    plt.ylabel("Bubble Count [-]")
    plt.grid(True, linestyle='--', alpha=0.75)

    # save the results
    os.makedirs(outputPath, exist_ok=True)

    figurePath = os.path.join(outputPath, 'bubbleCount.png')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()

def plotAvgBubbleDia(bubbleDiams, outputPath, dt):
    """
    Plots the average bubble diameter over time and saves the plot as an image.

    Parameters:
    bubbleDiams (list of float): List of bubble diameters at each time point.
    outputPath (str): Directory path where the plot image will be saved.
    dt (float): Time interval between each measurement in minutes.

    Returns:
    None
    """
    # generate time points for bubble sizes
    timePoints = [i * dt for i in range(len(bubbleDiams))]

    # plot average bubble size
    plt.figure(figsize=(12,8))
    plt.scatter(timePoints, bubbleDiams, color='green')
    plt.title('Average Bubble Diameter over time')
    plt.xlabel('Time [min]')
    plt.ylabel('D(t) [pixels]')

    # save the results
    os.makedirs(outputPath, exist_ok=True)

    figurePath = os.path.join(outputPath, 'avgBubbleDiams.png')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()

def plotBubbleSizes(bubbleSizes, idx, outputPath, dt):
    """
    Plots the bubble sizes over time and saves the plot as an image file.
    
    Args:
        bubbleSizes (list of float): A list containing the sizes of the bubbles.
        idx (int): Iteration index
        outputPath (str): Output path where the file should be stored.
        dt (float): The time interval between each bubble size measurement.
    
    Returns:
        None
    """
    # convert to numpy array and sort
    bubbleSizes = np.array(bubbleSizes)
    bubbleSizes = np.sort(bubbleSizes)

    # generate time points for bubble sizes
    timePoints = [i * dt for i in range(len(bubbleSizes))]

    # color per bubble
    colors = cm.jet(np.linspace(0, 1, len(bubbleSizes)))

    # plotting
    plt.figure(figsize=(12,8))

    for i in range(len(bubbleSizes)):
        plt.scatter(timePoints[i], bubbleSizes[i], color=colors[i], label=f"Bubble {i + 1}")

    plt.title("Bubble Size Over Time")
    plt.ylabel("Bubble Size (pixels²)")
    plt.grid(True, linestyle='--', alpha=0.75)

    # hide x axis
    plt.gca().get_xaxis().set_visible(False)

    if len(bubbleSizes) < 15:
        plt.legend()

    # save the results
    os.makedirs(outputPath, exist_ok=True)

    figurePath = os.path.join(outputPath, f'bubbleSizes_frame{idx:04d}.png')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()

    # save the sizes as csv file
    csvPath = os.path.join(outputPath, f'bubbleSizes_frame{idx:04d}.csv')
    np.savetxt(csvPath, bubbleSizes, delimiter=',', header='Bubble Sizes (pixels²)')

def loadAnnotations(annotationsPath, idx):
    """
    Load annotations from XML files in the specified directory.

    This function reads all XML files in the given directory, parses them,
    and extracts bounding box coordinates for each object. The coordinates
    are returned as a list of lists, where each inner list contains four
    float values representing the bounding box: [xmin, ymin, xmax, ymax].

    Args:
        annotationsPath (str): The path to the directory containing the XML annotation files.
        idx (int): Index for reading correct file.

    Returns:
        list: A list of bounding box coordinates. Each element in the list is a list of four floats: [xmin, ymin, xmax, ymax].
    """
    bndboxes = []

    # read the xml file
    for file in os.listdir(annotationsPath):
        if file.endswith(f'{idx + 1}.xml'):
            filePath = os.path.join(annotationsPath, file)
            tree = ET.parse(filePath)
            root = tree.getroot()

            for obj in root.findall('object'):
                # gather bnd box data
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                coordinates = [xmin, ymin, xmax, ymax]
                bndboxes.append(coordinates)

    return bndboxes

def calculateIOU(detections, annotations, idx, resultsPath):
    """
    Calculate the Intersection over Union (IoU) between detected boxes and annotation boxes.

    Args:
        detections (list): A list of detection results, where each detection is a list or tuple
            containing at least four elements representing the bounding box coordinates [x1, y1, x2, y2].
        annotations (list): A list of the annotations containing the xmin, ymin, xmax, ymax values
        idx (int): Index of the current frame processed.
        resulsPath (str): Output path to store results

    Returns:
        np.ndarray: A 2D array containing the IoU values between each pair of detected and annotation boxes.
    """
    # extract detected and annotation boxes and convert to torch tensor
    detBoxes = torch.tensor([det[:4] for det in detections], dtype=torch.float32)
    annBoxes = torch.tensor(annotations, dtype=torch.float32)

    # get the iou values
    iouMatrix = box_iou(detBoxes, annBoxes)

    # convert to numpy array for clean-up
    iouMatrix = iouMatrix.numpy()
    threshold = 0.5 # literature uses 0.5 or 0.75, for this simulation 0.5 is used
    iouMatrix[iouMatrix < threshold] = 0.0

    nonZeros = ~np.all(iouMatrix == 0, axis=0)
    updatedMatrix = iouMatrix[:, nonZeros]

    # generate output path for textfile
    os.makedirs(resultsPath, exist_ok=True)
    iouFile = f'iouMatrix_frame{idx:04d}.txt'
    outputPath = os.path.join(resultsPath, iouFile)

    # write IoU matrix to a text file
    np.savetxt(outputPath, updatedMatrix, delimiter=',', header="IoU Matrix")

    return updatedMatrix

def makeGif(directory):
    """
    Creates an animated GIF from a sequence of PNG images in the specified directory.
    
    Args:
        directory (str): The path to the directory containing the PNG images.
    
    Returns:
        None
    """
    try:
        files = [file for file in os.listdir(directory) if file.endswith('.png')]
        files.sort()

        images = []

        for file in files:
            path = os.path.join(directory, file)
            image = iio.imread(path)
            images.append(image)

        outputPath = os.path.join(directory, "tracking.gif")

        iio.imwrite(outputPath, images, duration=100, loop=0)

    except Exception as e:
        print(f"Failed to make gif: {e}")

def cleanDirectory(directories):
    """
    Function to ensure the results folder only contains outputs from the current simulation.
    
    Args:
        directories (lst): List of input paths.

    Returns:
        None
    """
    try:
        for directory in directories:
            if os.path.isfile(directory):

                # change file permission to allow writing
                os.chmod(directory, 0o777) # grants read/write/execute permissions
                os.unlink(directory)

            elif os.path.isdir(directory):

                # change file permission to allow writing
                os.chmod(directory, 0o777) # grants read/write/execute permissions
                shutil.rmtree(directory)

    except Exception as e:
        print(f"Failed to delete {directory}: {e}")