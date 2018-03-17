import glob
import inspect
import re
from collections import deque

import cv2
import numpy as np
import time
from scipy.ndimage.measurements import label

from Classifier import Classifier


class FramesetProcessor:
    slwindows = None
    def __init__(self, classifier, predict_threshold=0.6, heat_threshold=10):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)



    def find_cars(self, frame):
        draw_img = np.copy(frame)
        subimgs = []
        bboxes = []
        for window in self.slwindows:
            xb, yb, hc, size = window

            if yb > 650:
                continue

            # processing lower-right quarter of the frame
            imgr = cv2.resize(frame[int(yb - size):yb, int(hc + xb):int(hc + xb + size)], (64, 64))
            subimgs.append(imgr)
            bboxes.append([(hc + xb, int(yb - size)), (int(hc + xb + size), yb)])

            # (not) processing lower-right quarter of the frame (as suggested by my session instructor as systems are too slow to run)
            # imgl = cv2.resize(img[int(yb - size):yb, int(hc - xb - size):int(hc - xb)], (64, 64))
            # subimgs.append(imgl)
            # bboxes.append([(hc - xb, int(yb - size)), (int(hc - xb - size), yb)])

        test_prediction = self.classifier.classify(subimgs)

        found = []

        for j in range(len(test_prediction[0])):
            if (test_prediction[0][j] == 1):
                # print(test_prediction[1][j])
                if test_prediction[1][j] > self.predict_threshold:
                    bbox = bboxes[j]
                    found.append((bbox[0], bbox[1]))
                    cv2.rectangle(draw_img, bbox[0],
                                  bbox[1], (255, 0, 255), 2)

        return found

    def process_frame_bgr(self, frame, visualize=False):
        if self.slwindows is None:
            self.slwindows = FramesetProcessor.sliding_windows(frame)
            self.past_heats = deque(maxlen=20)

        #find windows containing cars
        found = self.find_cars(frame)
        heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heatmap = FramesetProcessor.add_heat(heatmap, found)
        self.past_heats.append(heatmap)
        heatmap = np.array(self.past_heats).sum(axis=0)
        # Apply threshold to help remove false positives
        heatmap = FramesetProcessor.apply_threshold(heatmap, self.heat_threshold)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        frame = FramesetProcessor.draw_labeled_bboxes(frame, labels)
        return frame

    def process_frame_rgb(self, frame, visualize=False):
        return cv2.cvtColor(self.process_frame_bgr(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),cv2.COLOR_RGB2BGR, visualize)


    @staticmethod
    def sliding_windows(img, ystart=475, ystop=650, yshift=37, size=64):
        increase = int(size / 2)
        width = img.shape[1]
        windows = []
        for yb in range(ystart, ystop, yshift):
            if (yb > ystop):
                break
            for xb in range(0, int(width / 2 - size * (1 - 0.25)), int(size * 0.25)):
                if ((width / 2 + xb + size) > width):
                    continue
                windows.append([xb, yb, int(width / 2), size])
            size += increase
        return windows


    @staticmethod
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    @staticmethod
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    @staticmethod
    def sorted_aphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    @staticmethod
    def load_frames(load_path):
        return FramesetProcessor.sorted_aphanumeric(glob.glob(load_path + "/*"))

if __name__ == '__main__':
    pass