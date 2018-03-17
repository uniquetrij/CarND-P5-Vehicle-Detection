import inspect

import cv2
import numpy as np
from skimage.feature import hog

from ColorComponents import ColorComponents


class HOGExtractor:
    visualization = None
    def __init__(self, cspace='HLS', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL", vis=False,
                 feature_vec=True):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract(self, feature_image, ccomponents=None):
        self.visualization = []
        if ccomponents is None:
            ccomponents = ColorComponents(feature_image)
        self.ccomponents = ccomponents
        feature_image = ccomponents.getSpace(self.cspace.lower())
        # Call get_hog_features() with vis=False, feature_vec=True
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                features = HOGExtractor.get_hog_features(feature_image[:, :, channel],
                                                                  self.orient, self.pix_per_cell, self.cell_per_block,
                                                                  vis=self.vis, feature_vec=self.feature_vec)
                if self.vis == True:
                    self.visualization.append(features[1])
                    features = features[0]
                hog_features.append(features)

            if self.feature_vec == True:
                hog_features = np.ravel(hog_features)
        else:
            hog_features = HOGExtractor.get_hog_features(feature_image[:, :, self.hog_channel], self.orient,
                                                         self.pix_per_cell, self.cell_per_block, vis=self.vis,
                                                         feature_vec=self.feature_vec)
            if self.vis == True:
                self.visualization.append(hog_features[1])
                hog_features = hog_features[0]

        # Append the new feature vector to the features list
        return hog_features

    # Define a function to return HOG features and visualization
    @staticmethod
    def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features


if __name__ == '__main__':
    from Dataset import Dataset
    dataset = Dataset.from_pickle("./dataset/dataset.p")
    # car = cv2.imread(dataset.cars[10])
    car = cv2.imread(dataset.notcars[50])
    img = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
    import matplotlib.pyplot as plt

    extractor = HOGExtractor(cspace="HLS",vis=True,feature_vec=False)
    extractor.extract(car)

    img = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)


    fig = plt.figure(figsize=(16, 4))
    plt.subplot(141)
    plt.imshow(img)
    plt.title('NOT CAR')
    plt.subplot(142)
    plt.imshow( extractor.visualization[0])
    plt.title('HOG HLS/H')
    plt.subplot(143)
    plt.imshow(extractor.visualization[1])
    plt.title('HOG HLS/L')
    plt.subplot(144)
    plt.imshow(extractor.visualization[2])
    plt.title('HOG HLS/S')
    fig.tight_layout()
    plt.show()


    # print(extractor.visualization[1].shape)
    # cv2.imwrite("./output_images/notcar.jpg", car)
    # cv2.imwrite("./output_images/notcar_hog_h.jpg", extractor.visualization[0]*255)
    # cv2.imwrite("./output_images/notcar_hog_l.jpg", extractor.visualization[1]*255)
    # cv2.imwrite("./output_images/notcar_hog_s.jpg", extractor.visualization[2]*255)



