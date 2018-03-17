## Writeup
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1.1]: ./output_images/car_hog.jpg
[image1.2]: ./output_images/notcar_hog.jpg

[image2.1]: ./output_images/car_hist.jpg
[image2.2]: ./output_images/notcar_hist.jpg

[image3.1]: ./output_images/windows.jpg
[image3.2]: ./output_images/windows_full.jpg

[image4.1]: ./output_images/test1_bb.jpg
[image4.2]: ./output_images/test3_bb.jpg


[image5.1]: ./output_images/f3.png
[image5.2]: ./output_images/f4.png
[image5.3]: ./output_images/f5.png
[image5.4]: ./output_images/f6.png
[image5.5]: ./output_images/f7.png
[image5.6]: ./output_images/f8.png

[image6.1]: ./output_images/h3.png
[image6.2]: ./output_images/h4.png
[image6.3]: ./output_images/h5.png
[image6.4]: ./output_images/h6.png
[image6.5]: ./output_images/h7.png
[image6.6]: ./output_images/h8.png

[image7.1]: ./output_images/l8.png
[image7.2]: ./output_images/b8.png

[video1]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it [Here](https://github.com/uniquetrij/CarND-P5-Vehicle-Detection/blob/master/writeup.md) !

First, I created a class `Dataset` (file named `Dataset.py`) that serves as a helper for loading the `vehicle` and `non-vehicle` images from a file system path. An instance of this class will serve as input to my `Classifier` mentioned later. An instance, say `dataset` of `Dataset` can be obtained in either of the following ways:
1. by passing the list of cars and not-cars image paths to the class initializer (code lines 7 through 11).

2. from the base system path containing images within directories labeled as `vehicles` and `non-vehicles` If a pickle file path is provided, it automatically  saves the dataset paths into the pickle file (code lines 20 through 45).
    
3. loading the dataset from a pickle file previously saved (code lines 14 through 17).

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Histogram Of Gradients (HOG) features of an image can be obtained using an instance of the class `HOGExtractor` in file named `HOGExtractor.py`. The class initializer takes the following arguments:

|Parameter              |Required/Default     |
|:---------------------:|:-------------------:| 
|cspace                 |'HLS'                |
|orient                 |9                    |
|pix_per_cell           |8                    |
|cell_per_block         |2                    |
|hog_channel            |'ALL'                |

Once the extractor is initialized with the required arguments, the hog features of an image can be obtained invoking the `extract()` method (code lines 21 through 37). This method internally calls the `get_hog_features()` method (code lines 41 through 56) to obtain the features for each color channels, combines it and returns the result. The `get_hog_features()` function, in turn, calls skimage's `hog()` method to calculate the HOG features.

#### 2. Explain how you settled on your final choice of HOG parameters.

I played with different parameter combinations (`orientations`, `pixels_per_cell`,and `cells_per_block`) and different color spaces (`RGB`, `YCrCb`, `YUV` etc) and finally settled upon using the default values mentioned in the table above. Those are the values which I found to work out best in my scenario of vehicle detection. Following is an example result of the  `HOGExtractor`

|Car HOG|
|:---------------------:|
|![alt text][image1.1]  |

|Not-Car HOG|
|:---------------------:|
|![alt text][image1.2]  |



I also created another two extractor classes named `SpacialExtractor` and `HistExtractor` (files by their respective names) to extract spacial bin features and color histogram features respectively.

|HistExtractor init Parameters|Required/Default     |
|:---------------------------:|:-------------------:| 
|cspace                       |'YUV'                |

Following is the `HistExtractor` extracted features for the car example above

|HistExtractor Features (Car)      |
|:--------------------------------:|
|![alt text][image2.1]             |

|HistExtractor Features (Not Car)      |
|:------------------------------------:|
|![alt text][image2.2]                 |


|SpacialExtractor init Parameter|Required/Default     |
|:-----------------------------:|:-------------------:| 
|cspace                         |'YUV'                |
|size                           |(32,32)              |

Since the extracted features needs to be combined together into a feature vector, I created a dedicated class named `FeatureCombiner` to perform this operation  (file `FeatureCombiner.py`). An instance, say `combiner` of `FeatureCombiner` can be obtained in either of the following ways:
1. by passing a list of extractors to the class initializer (code lines 14 through 22). If a pickle file path is provided, it automatically  saves the instance along with the extractors list, so that it can be reused.

2. loading an instance from a pickle file previously saved (code lines 25 through 28).

The method `combiner.from_dataset()` can be used to prepare a `dataset`. This method automatically extracts the features from each data instance (image) in the dataset (as per the list of extractors provided during initialization), combines them, and returns the featureset. Note that, since dataset contains both car and non-car images, this method returns two list of feature vectors respectively. This method comes handy during training a `classifier` from a `dataset`.

Similarly, the other method `combiner.from_images()` extracts features from a list of images processing the images in the same way as the method above. This method comes handy while predicting a set of images, like sub-images from a video frame.

Since both images for training/testing a classifier and images presented to the classifier for prediction must go through the same preprocessing to obtain the feature vectors, the above two methods can be directly used by a classifier implicitly, without the user having to write the preprocessing statements again and again.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I designed a class named `Classifier` (file `Classifier.py`) that can be used to train an SVM classifier. An instance say `classifier` of `Classifier` can be obtained by passing a `combiner` to its class initializer (code lines 19 through 23). This `classifier` then can be trained using `classifier.train()` method (code lines 31 through 71). The `train()` method takes a `dataset` as an argument and trains a 'linear' SVM. To do so, it first passes the dataset through the `combiner` to obtain the featuresets of car and noncar data, splits it into train and test set in 4:1 ratio, scales the features using `sklearn.StandardScaler()` and then trains an `sklearn.svm.SVC` model. If a pickle file path is provided, it automatically  saves the trained svm model into the pickle file, which can be later loaded using the `from_pickle()` method of the class. 

Once the model is trained, the other method of the class viz. `classify()` (code lines 73 through 77) can be used to predict on a list of images. This method returns the predicted class and its decision function value for each image in the list.

The training of the `classifier` was done as follows:
```python

    dataset = Dataset.from_path(max_size=4000, pickle_path="./dataset/dataset.p")
    combiner = FeatureCombiner([SpacialExtractor(), HistExtractor(), HOGExtractor()],
                               pickle_path="./dataset/combiner.p")
    classifier = Classifier(combiner)
    classifier.train(dataset, pickle_path="./dataset/classifier.p")

```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since the objective of the project is to identify vehicles, it is judicious to ignore the upper half of the image and concentrate on the lower half only. Let's call it our region of interest or `roi`. It can be observed that the higher an object is in the `roi`, the farther it is in the scene, and the smaller it appears in the roi. In short, it follows a linear perspective. Thus, I planned to scale my window linearly as the base of my sliding window moved down in `Y` direction. I chose to window the frames in between y_start=475 and y_stop=650 with a vertical shift of 37. For the overlap in the horizontal `X` direction, I chose it to be 25%. I chose the minimum window size to be 64x64 with an increment of 32 pixels in either direction on each vertical shift.

Here is an example showing the perspective of two sliding windows along `Y` direction.

|Example Sliding Window |Full Range Sliding Window |
|:---------------------:|:------------------------:|
|![alt text][image3.1]  |![alt text][image3.2]     |

The `FramesetProcessor` class in file `FramesetProcessor.py` contains all such methods that are required to process each image frame. This class can be instantiated by passing a trained `classifier` to its initializer method. The `sliding_windows()` method (code lines 84 through 96) returns the list of all window coordinates that can be used to subsample the frame. The `find_cars()` method (code lines 25 through 47) uses these windows to clip a sub-image from the frame and then checks it with the `classifier` whether that patch contains a car. If yes, it generates a list of "found" windows and returns it. These than then be used to generate heat maps, smoothen, and annotate the image (method `process_frame_bgr()`, code lines 60 through 77). Note that that method expects a "BGR" image. Alternatively the `process_frame_rgb()` automatically does the conversion and internally calls `process_frame_bgr()` and re-transforms the returned frame.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned YUV and histograms of YUV in the feature vector, which provided a nice result.  Here are some example images:

|Example 1              |Example 2            |
|:---------------------:|:-------------------:| 
|![alt text][image4.1]  |![alt text][image4.2]|
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/uniquetrij/CarND-P5-Vehicle-Detection/blob/master/project_video_annotated.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The `process_frame_rgb()` method of `FramesetProcessor` internally creates a heatmap from the overlapping "found" windows. This heat represents the confidence of a car being present at that location in the frame. Here I also maintained a queue of past 20 heatmaps which allowed me to sum them up and then apply a threshold to identify vehicle positions more correctly with much smoother bounding boxes across frames. This also eliminates chances of false positives being too frequent. Finally I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

|Frames                 |Heatmaps            |
|:---------------------:|:-------------------:| 
|![alt text][image5.1]  |![alt text][image6.1]|
|![alt text][image5.2]  |![alt text][image6.2]|
|![alt text][image5.3]  |![alt text][image6.3]|
|![alt text][image5.4]  |![alt text][image6.4]|
|![alt text][image5.5]  |![alt text][image6.5]|
|![alt text][image5.6]  |![alt text][image6.6]|

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image7.1]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7.2]


The `Main.py` file contains a method named `process_video()` which takes the video file path and the path where the annotated video will be saved. This method internally calls the `process_frame_rgb()` of `FramesetProcessor` as the video frames passed are in RGB format. This method processes the entire video and saves the annotated video in the mentioned location.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The major drawback I faced was the time it was taking to train the model and process the video, especially with a limited hardware. Moreover, I believe unlike tensorflow, scikit-learn don't use GPU to speedup processing. This is probably because neural networks are truly parallel in nature, while SVM and other classifiers are not.

2. As a result, tuning parameters were quite difficult. while one combination of parameters worked well in one section of the video, it might not work efficiently on another section. It was quite a task to reach some sub-optimal tuning.

3. To achieve some speedup, our session instructor advised us to only concentrate on the lower-right quarter of the image instead of the entire lower half. 

4. Although HOG-Subsampling could have been more efficient, it produced jittery bounding boxes, more often false positives, and also sometimes false negatives (missing the car entirely for few consecutive frames). Therefore I used simple sliding window technique.

5. Performance can be improved using an 'rbf' kernel instead of 'linear' one. Due to system constraints, I wasn't able to try 'rbf' as it was taking too long to train.

6. Convolution and Deep Neural Network based classification would surely enhance performance as those can be executed on a fast GPU thereby making the classification response more real-time. YOLO might be interesting to look into in this regard.

