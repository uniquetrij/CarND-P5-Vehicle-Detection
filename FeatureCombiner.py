import inspect
import pickle
import cv2
import numpy as np

from ColorComponents import ColorComponents
from HOGExtractor import HOGExtractor
from HistExtractor import HistExtractor
from SpacialExtractor import SpacialExtractor


class FeatureCombiner:

    def __init__(self, extractors, pickle_path=None):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

        if pickle_path is not None:
            combiner = {"combiner": self}
            pickle.dump(combiner, open(pickle_path, "wb"))

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, mode='rb') as f:
            combiner = pickle.load(f)
        return combiner['combiner']

    def from_dataset(self, dataset):
        cars = dataset.cars
        notcars = dataset.notcars
        cars_features = self.from_files(cars)
        notcars_features = self.from_files(notcars)

        return cars_features, notcars_features

    def from_files(self, files):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in files:
            image = cv2.imread(file)
            features.append(FeatureCombiner.combine(image, self.extractors))
        # Return list of feature vectors
        return features

    def from_images(self, images):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for image in images:
            features.append(FeatureCombiner.combine(image, self.extractors))
        # Return list of feature vectors
        return features

    @classmethod
    def combine(cls, img, extractors):
        ccomponents = ColorComponents(img)
        extracts = []
        for extractor in extractors:
            extracts.append(cls.extract(ccomponents, extractor))
        extracts = np.concatenate(extracts)
        return extracts

    @classmethod
    def extract(cls, ccomponents, extractor):
        feature = extractor.extract(None, ccomponents)
        return feature


if __name__ == '__main__':
    combiner = FeatureCombiner([SpacialExtractor(), HistExtractor(), HOGExtractor()],
                               pickle_path="./dataset/combiner.p")
