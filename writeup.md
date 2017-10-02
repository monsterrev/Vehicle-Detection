##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/car_car.png
[image2]: ./output_images/car_not_car.png
[image3]: ./output_images/HOG_example.png
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/sliding_window.jpg
[image6]: ./output_images/bboxes_and_heat.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/heat_map1.png
[image9]: ./output_images/heat_map2.png
[image10]: ./output_images/last_frame.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
 
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
The code for this step is contained in the first code cell of the IPython notebook test.ipynb in the first cell. A random sample of images is shown below

![alt text][image1]

Not car images
![alt text][image2]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The code is defined in section Hog Features and the function name is get_hog_features()

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image3]

After this I defined functions to extract color features in the section Color Features, e..g functions are convert_color which concerts image from one colorspace to another. bin_spatial computes the binned color feature, color_hist to computer histogram features.

####2. Explain how you settled on your final choice of HOG parameters.

Extract All Features contains a function which helps us extract all the features, we can just pass in the booleans and it will return all those features. Spatial, Hog and histogram. 

I tried various combinations of parameters and I settled on HOG parameters based upon the accuracy and performance of the SVM classifier built using them. Increasing the number of orientation or number of bins would have increased the number of features by a lot and a balance was required so as to avoid overfitting of the model. Sometimes it would happen that we will get 100% accuracy but the time to compute that would also increase manifold time compromising the performance. A balanced approach balancing accuracy and performance is chosen and final parameters are:
colorspace : 'YUV'
Orientations: 11
Pixels per cell: 16
Cells per block: 2


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using featues extracted from the images as described in above section. The code for training is present in the section "Training the classifier". First I defined all the parameters required for the extract_features function and then extracted the features for car and non car images. 
Since we are dealing with image data that are going to be extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. 
While it is possible to achieve a sufficiently good result on the project without worrying about time-series issues, but to optimize the classifier I have used train/test split that avoids having nearly identical images in both your training and test sets. This means extracting the time-series tracks and separating the images manually to make sure train and test images are sufficiently different from one another.
I just used HOG feature set excluding spatial_bin and histogram features and gained following performance:

Using: 11 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1188
0.03 Seconds to train SVC...
Test Accuracy of SVC =  0.99

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Running the search at random position and at random scale all over the image would have been the easiest solution but it is highly inefficient. I tried a couple of approached as described in the lesson and implemented in section Sliding Window and some optimizations are made as well like we dont search the entire image but only the portions where we would find the cars (y_start 400 to y_stop=700 and this approach seemed to work decently. Below is an example for the same 
![alt text][image4]

  The code is taken from the class and function name is search_windows and slideWindow, where for each window scale and the rectangles returned from each method call are aggregated. Small scale of .5 were giving too many false positives and playing around a bit with overlap helped in improving a bit. Additionally, only an appropriate vertical range of the image is considered for each window size (e.g. smaller range for smaller scales) to reduce the chance for false positives in areas where cars at that scale are unlikely to appear. 

The image below shows the rectangles returned by find_cars drawn onto one of the test images in the final implementation. Notice that there are several positive predictions on each of the near-field cars, and one positive prediction on a car in the oncoming lane.

![alt text][image5]

This is obtained after elimintaing the false positives.
Since there were many false detections and false positives, we use the approach as defined in the class, i.e. To make a heat-map, we simply  add "heat" (+=1) for all pixels within windows where a positive detection is reported by your classifier. The individual heat-maps for the above image look like this and the corresponding image.
![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image7]

Some of the optimizations:
1. Instead of searching through the entire image, I would just take portion of image where the images are likely to be present. Since the camera is mounted at the center of the car, the car would be seen only in the portion 300-700 (height). It cant be present on the top of tree so no point searching for it in that area. 
2. Though as we increase the number of features it is likely that our model would perform better. But this puts a significant cost on the performance and hence I have not included spatial and hist_bin features. This also helps in prevention of overfitting of the model. 


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  I tried to use other features like hist and spatial_bin, though they increased the accuracy of training set but performed poorly with the example test images and there were lot of false positives and false detections even after using the heat map. So ultimately I switched those off. This also helped in improving the performance. 
A lot of effort was needed to choose the parameters such that we get good results. Event though my implementation have few false positives but I believe it is better than not detecting the car which would be worse and would likely cause accidents. But I guess my implementation makes it extra cautious. 
Pipeline does not work best in cases - 
## frames dont resemble training dataset. - Empty image also detects something.
## Lighting and environment condition. white car in white background or detection of car when there is snowfall would be tricky.

To improve the model, probably we can use the neural networks or better CNN as read in previous classes. Hope to improve the model and get better results using that.


