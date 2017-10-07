**Vehicle Detection Project**

Udacity designed the project for using classic Computer Vision techniques, namely HOG features and SVM classifier. However they allowed the flexiblity to use any machine learning or deep learning apparoch to detect vehicals. 

In my prioir hobby projects, I've evaluated both HOG/SVM and Tensorflow/Yolo. I've come to the conclusion that, if you have a GPU availalbe, then Yolo runs faster than SVM and provides detection of 80 different classes (10 of which are of interest in automotive). A properly traiend HOG/SVM can match or maybe exceed the accuracy of a deep learning approach but it only does that for a single class of objects! On the road, a self driving car, would encounter many unforeseen scenarios. For example if you trained your HOG/SVM on only cars, then what will happen if it encounters atypical vehicle class, like the ones used in construction and agriculture? HOG/SVM has its place when you dont have lot of training data or scope of your search is constrained. 

A Deep Learning approach can cope better with wide range of objects encountered on the road. As compared to a carefully hand crafted HOG/SVM. 

[//]: # (Image References)
[image1]: ./output_images/out2.jpg
[image2]: ./output_images/out5.jpg
[image3]: ./output_images/out8.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

#### Test Run

The follwing image is result of my Tensorflow/Yolo pipleline. Please notice it detects pedestrians, traffic lights in addition to cars. If I use a differnt model, VOC vs COCO, I get bus detected but lose traffic lights.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Model Details
#### Architecture

| Layer description                | Output size|
|:---------------------------------|:--------------|
| input                            | (608, 608, 3)|
| conv 3x3p1_1                     | (608, 608, 32)|
| maxp 2x2p0_2                     | (304, 304, 32)|
| conv 3x3p1_1                     | (304, 304, 64)|
| maxp 2x2p0_2                     | (152, 152, 64)|
| conv 3x3p1_1                     | (152, 152, 128)|
| conv 1x1p0_1                     | (152, 152, 64)|
| conv 3x3p1_1                     | (152, 152, 128)|
| maxp 2x2p0_2                     | (76, 76, 128)|
| conv 3x3p1_1                     | (76, 76, 256)|
| conv 1x1p0_1                     | (76, 76, 128)|
| conv 3x3p1_1                     | (76, 76, 256)|
| maxp 2x2p0_2                     | (38, 38, 256)|
| conv 3x3p1_1                     | (38, 38, 512)|
| conv 1x1p0_1                     | (38, 38, 256)|
| conv 3x3p1_1                     | (38, 38, 512)|
| conv 1x1p0_1                     | (38, 38, 256)|
| conv 3x3p1_1                     | (38, 38, 512)|
| maxp 2x2p0_2                     | (19, 19, 512)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| conv 1x1p0_1                     | (19, 19, 512)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| conv 1x1p0_1                     | (19, 19, 512)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| concat [16]                      | (38, 38, 512)|
| conv 1x1p0_1                     | (38, 38, 64)|
| local flatten 2x2                | (19, 19, 256)|
| concat [27, 24]                  | (19, 19, 1280)|
| conv 3x3p1_1                     | (19, 19, 1024)|
| conv 1x1p0_1                     | (19, 19, 425)|

Leaky ReLU follows all convolution layers except the last one which is "linear".

#### Implementation
I've used two differnt implenetation of YOLO with TensorFlow https://github.com/thtrieu/darkflow and https://github.com/allanzelener/YAD2K. I've built abstraction that both can be used interchangeably. I wanted to evualute accuracy vs speed. Both were compareable.




### Video Implementation

Here's a [link to my video result](./project_video_output.mp4)

#### Pipeline
The first thing I do is to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result. I've observed that undistored images worke better with YOLO and I was able to detect smaller objects, like cars on the opposite lane!

I feed the undistored image to a Lane finder that I implemented earlier using Classic Computer Vision techniques of color transforms, and gradient thresholding. It identifies lane curvature and vehicle displacement and is robust against environmental challenges such as shadows and pavement changes.

Finally output is annotated by both the Lane Finder and Yolo tracker.

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Camera distortion seems to affect Tensorflow/Yolo detection. It sometimes confuses car and truck which is not too terrible. Deep learning approach works great if you have lot of data to train it or you can reuse pre-trained network with transfer learning. It does require GPU in order to run in realtime. 

SVM/HOG also works well but the detection scope is limited to the class you trined it for.  On the road, a self driving car, would encounter many unforeseen scenarios. For example if you trained your HOG/SVM on only cars, then it may fail if it encounters atypical vehicle class, like the ones used in construction and agriculture.

SVM with sliding window does not run realtime on modern CPUs. However if you have FPGA option available on your hardware platform then acceleration is possible. Both training and accelration of HOG/SVM requires more upfront engineering effort. 

