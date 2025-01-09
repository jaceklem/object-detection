import torch
import numpy as np
from scipy.spatial import cKDTree
from torchvision.ops import nms

from extendedKF import ExtendedKalmanFilter

class MultiObjectTracking:
    """
    A class to manage multiple Kalman Filters for multi-object tracking
    """
    def __init__(self, stateDim, measDim, processNoiseCov, measNoiseCov, confThreshold, bubbleGrowth, iouThresh, maxDist, debugMode):
        self.trackers = []

        self.stateDim = stateDim
        self.measDim = measDim
        self.processNoiseCov = processNoiseCov
        self.measNoiseCov = measNoiseCov
        self.confThreshold = confThreshold
        self.bubbleGrowth = bubbleGrowth

        self.iouThresh = iouThresh
        self.maxDist = maxDist
        self.debugMode = debugMode

    def update(self, detections, idx):
        """
        Updates each Kalman filter based on new detections.
        Only the tracker with the highest confidence score is kept.

        Args:
            detections (list): List of detected centers from YOLO.
        """

        # remove low confidence detections
        detections = [det for det in detections if det[2] >= self.confThreshold]

        if not detections:
            print("No valid detections passed.")
            return 
        
        detections.sort(key=lambda x: x[2], reverse=True)

        # apply nms
        detections = self.nonMaxSuppression(detections)
        
        predPositions = [tracker.x[:2] for tracker in self.trackers]
        matchedIndices, unmatchedIndices, unmatchedTrackers = self.matchDetections(detections, predPositions)
        
        # extra information
        if self.debugMode:
            self.debugPrint(detections, matchedIndices, unmatchedIndices, unmatchedTrackers, predPositions)

        updatedTrackers = {}

        for detIdx, trackerIdx in matchedIndices:
            detection = detections[detIdx]
            detectedCenter, confidence = detection[:2], detection[2]

            # compare confidence scores
            if trackerIdx not in updatedTrackers or confidence > updatedTrackers[trackerIdx][1]:
                updatedTrackers[trackerIdx] = (detectedCenter, confidence)

        # update tracker with highest confidence score
        for trackerIdx, (detectedCenter, confidence) in updatedTrackers.items():
            self.trackers[trackerIdx].predict(u=np.zeros((self.stateDim, 1)))
            self.trackers[trackerIdx].update(detectedCenter)
            self.trackers[trackerIdx].confidence = confidence

            if self.debugMode:
                print(f"updated tracker {trackerIdx} with detection {detections[detIdx]}")

        # add new tracker for unmatched detections
        for detIdx in unmatchedTrackers:
            newTracker = ExtendedKalmanFilter(self.stateDim, self.measDim, self.processNoiseCov, self.measNoiseCov)
            newTracker.x[:2] = detections[detIdx][:2] # position
            newTracker.x[4] = 1.0 # initial size
            newTracker.confidence = detections[detIdx][2] # confidence
            self.trackers.append(newTracker)
            
            if self.debugMode:
                print(f"new tracker for detection {detections[detIdx]}")

        # delete unused trackers
        self.trackers = [tracker for idx, tracker in enumerate(self.trackers) if idx not in unmatchedTrackers]  

        # remove low-confidence trackers
        self.remLowConfTrackers()

    def matchDetections(self, detections, predictions):
        """
        Matches detections to trackers based on nearest neighbor distances.

        Args:
            detections (list): List of detected centers from YOLO.
            predictions (list): List of predicted positions from Kalman filters.

        Returns:
            matched_indices (list): List of matched (detection index, tracker index) pairs.
            unmatched_detections (list): List of detection indices with no match.
            unmatched_trackers (list): List of tracker indices with no match.
        """
        # checks if lists are empty
        if not detections or not predictions:
            return [], list(range(len(detections))), list(range(len(predictions)))

        # k-d tree for nearest-nearest neighboring
        tree = cKDTree(predictions)
        k = max(1, min(int(0.25 * len(predictions)), len(predictions)))
        distances, indices = tree.query(detections, distance_upper_bound=self.maxDist, k=k)
 
        matchedIndices = []
        unmatchedDetections = set(range(len(detections)))
        unmatchedTrackers = set(range(len(predictions)))

        # find the matches
        for detIdx, minDist in enumerate(distances):
            minIdx = indices[detIdx]

            if minDist < self.maxDist:
                matchedIndices.append((detIdx, minIdx))
                unmatchedDetections.discard(detIdx)
                unmatchedTrackers.discard(minIdx)

        return matchedIndices, list(unmatchedDetections), list(unmatchedTrackers)
    
    def nonMaxSuppression(self, detections):
        """
        Perform Non-Maximum Suppression (NMS) on a list of detections to eliminate 
        redundant overlapping detections.
        
        Args:
            detections (list): A list of detections            
            overlapThresh (float): The threshold for considering two detections as overlapping. 
        
        Returns:
            nms (list): A list of detections after applying Non-Maximum Suppression.
        """
        if len(detections) == 0:
            return []
        
        # suppressed detections
        boxes = torch.tensor([det[:4] for det in detections], dtype=torch.float32) # x and y coordinates
        scores = torch.tensor([det[4] for det in detections], dtype=torch.float32) # confidence

        # filter detections
        keepIndices = nms(boxes, scores, self.iouThresh)
        suppressed = [detections[idx] for idx in keepIndices]

        return suppressed

    def remLowConfTrackers(self):
        """
        Removes trackers that have lower confidence scores than the threshold
        """
        # initialize set
        remove = set()

        for i, tracker1 in enumerate(self.trackers):
            # skips removable trackers
            if i in remove:
                continue
            
            # comparison check
            for j in range(i + 1, len(self.trackers)):
                tracker2 = self.trackers[j]

                # check if valid arrays
                if tracker1.x.size < 2 or tracker2.x.size < 2:
                    continue

                dist = np.linalg.norm(tracker1.x[:2] - tracker2.x[:2])

                if dist < self.maxDist:
                    # remove first tracker
                    remove.add(i if tracker1.confidence < tracker2.confidence else j)
        
        # update allowed trackers
        self.trackers = [tracker for idx, tracker in enumerate(self.trackers) if idx not in remove]
        
        if self.debugMode:
            print(f"Trackers marked for removal: {remove}")
            print(f"After removal: {[(tracker.x[:2], tracker.confidence) for tracker in self.trackers]}")

    def debugPrint(self, detections, matchedIndices, unmatchedIndices, unmatchedTrackers, predPositions):
            """
            Helper function for debug print statements.
            """
            print(f"Predicted positions: {predPositions}")
            print(f"Detections: {detections}")
            print(f"Matched indices: {matchedIndices}")
            print(f"Unmatched indices: {unmatchedIndices}")
            print(f"Unmatched trackers: {unmatchedTrackers}")