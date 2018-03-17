import glob
import inspect
import pickle
from random import shuffle

class Dataset:
    def __init__(self, cars, notcars):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, mode='rb') as f:
            dataset = pickle.load(f)
        return Dataset(dataset['cars'], dataset['notcars'])

    @classmethod
    def from_path(cls, globBasePath='./dataset', max_size = 4000, pickle_path=None):
        images = glob.glob(globBasePath + "/**/*.jpeg", recursive=True)
        # images = images + glob.glob(globBasePath+"/**/*.png", recursive=True)

        cars = []
        notcars = []
        for image in images:
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)
        print("Available Car Images ", len(cars))
        print("Available Non-Car Images ", len(notcars))
        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time
        cars = cars[0:max_size]
        notcars = notcars[0:max_size]
        shuffle(cars)
        shuffle(notcars)
        print("Using Subset Size ", len(cars), len(notcars))

        if pickle_path is not None:
            dataset = {"cars": cars, "notcars": notcars}
            pickle.dump(dataset, open(pickle_path, "wb"))

        return Dataset(cars, notcars)

    def data_stats(self):
        if self.datastats is None:
            data_dict = {}
            # Define a key in data_dict "n_cars" and store the number of car images
            data_dict["n_cars"] = len(self.cars)
            # Define a key "n_notcars" and store the number of notcar images
            data_dict["n_notcars"] = len(self.notcars)
            # Read in a test image, either car or notcar
            tmp_img = self.cars[0]
            # Define a key "image_shape" and store the test image shape 3-tuple
            data_dict["image_shape"] = tmp_img.shape
            # Define a key "data_type" and store the data type of the test image.
            data_dict["data_type"] = tmp_img.dtype
            # Return data_dict
            self.datastats = data_dict
        return self.datastats

if __name__ == '__main__':
    # dataset = Dataset.from_path(max_size=1000, pickle_path="./dataset/dataset.p")
    pass