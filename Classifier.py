import inspect

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

from Dataset import Dataset
from FeatureCombiner import FeatureCombiner
from HOGExtractor import HOGExtractor
from HistExtractor import HistExtractor
from SpacialExtractor import SpacialExtractor
import pickle


class Classifier:
    def __init__(self, combiner):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, mode='rb') as f:
            classifier = pickle.load(f)
        return classifier['classifier']

    def train(self, dataset, pickle_path=None):
        car_features, notcar_features = self.combiner.from_dataset(dataset)
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler only on the training data
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = SVC(kernel="linear", C=1)
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
        self.svc, self.X_scaler = svc, X_scaler

        if pickle_path is not None:
            classifier = {"classifier": self}
            pickle.dump(classifier, open(pickle_path, "wb"))
        return svc, X_scaler

    def classify(self, images):
        sample_features = self.combiner.from_images(images)
        X = np.array(sample_features).astype(np.float64)
        X = self.X_scaler.transform(X)
        return self.svc.predict(X), self.svc.decision_function(X)


if __name__ == '__main__':
    dataset = Dataset.from_pickle("./dataset/dataset.p")
    combiner = FeatureCombiner.from_pickle("./dataset/combiner.p")
    classifier = Classifier(combiner)
    classifier.train(dataset, pickle_path="./dataset/classifier.p")
    # classifier = Classifier.from_pickle("./dataset/classifier.p.old")
    # img = cv2.imread(Dataset.from_path(max_size=1).notcars[0])
    # print(classifier.classify([img]))
