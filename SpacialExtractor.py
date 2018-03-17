import inspect

import cv2

from ColorComponents import ColorComponents


class SpacialExtractor:
    spacial = None

    def __init__(self, cspace="YUV", size=(32, 32)):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

    # Define a function to compute binned color features
    def extract(self, img, ccomponents=None):
        if ccomponents is None:
            ccomponents = ColorComponents(img)
        img = ccomponents.getSpace(self.cspace.lower())
        # Use cv2.resize().ravel() to create the feature vector
        self.spacial = cv2.resize(img, self.size)
        features = self.spacial.ravel()
        # Return the feature vector
        return features


if __name__ == '__main__':
    from Dataset import Dataset

    dataset = Dataset.from_pickle("./dataset/dataset.p")
    car = cv2.imread(dataset.cars[10])
    notcar = cv2.imread(dataset.notcars[50])

    img = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt

    extractor = SpacialExtractor()
    extractor.extract(car)


    plt.imshow(extractor.spacial[:,:,0])
    plt.title('Spacial YUV')
    plt.show()
