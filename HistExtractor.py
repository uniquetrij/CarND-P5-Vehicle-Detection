import inspect

import cv2
import numpy as np

from ColorComponents import ColorComponents


class HistExtractor:
    histograms = None

    def __init__(self, cspace="YUV"):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

    # Define a function to compute binned color features
    def extract(self, img, ccomponents=None, nbins=32, bins_range=(0, 256)):
        self.histograms = []
        if ccomponents is None:
            ccomponents = ColorComponents(img)
        img = ccomponents.getSpace(self.cspace.lower())
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

        # Generating bin centers
        self.bin_edges = channel1_hist[1]
        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[0:len(self.bin_edges) - 1]) / 2

        # Concatenate the histograms into a single feature vector
        features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        self.histograms.append(channel1_hist[0])
        self.histograms.append(channel2_hist[0])
        self.histograms.append(channel3_hist[0])
        # Return the individual histograms, bin_centers and feature vector
        return features

if __name__ == '__main__':
    from Dataset import Dataset
    dataset = Dataset.from_pickle("./dataset/dataset.p")
    car = cv2.imread(dataset.cars[10])
    notcar = cv2.imread(dataset.notcars[50])

    import matplotlib.pyplot as plt

    extractor = HistExtractor()
    extractor.extract(notcar)


    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(extractor.bin_centers, extractor.histograms[0])
    plt.xlim(0, 256)
    plt.title('Y Histogram')
    plt.subplot(132)
    plt.bar(extractor.bin_centers, extractor.histograms[1])
    plt.xlim(0, 256)
    plt.title('U Histogram')
    plt.subplot(133)
    plt.bar(extractor.bin_centers, extractor.histograms[2])
    plt.xlim(0, 256)
    plt.title('V Histogram')
    fig.tight_layout()
    plt.show()


    # cv2.imwrite("./output_images/car.jpg", car)
    # cv2.imwrite("./output_images/car_hist_y.jpg", extractor.visualization[0])
    # cv2.imwrite("./output_images/car_hist_u.jpg", extractor.visualization[1])
    # cv2.imwrite("./output_images/car_hist_v.jpg", extractor.visualization[2])
    #
    # cv2.imwrite("./output_images/notcar.jpg", notcar)
    # cv2.imwrite("./output_images/notcar_hist_y.jpg", extractor.visualization[0])
    # cv2.imwrite("./output_images/notcar_hist_u.jpg", extractor.visualization[1])
    # cv2.imwrite("./output_images/notcar_hist_v.jpg", extractor.visualization[2])

