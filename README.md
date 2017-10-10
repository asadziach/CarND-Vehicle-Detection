**Vehicle Detection Project**

Udacity designed this project for using classic Computer Vision techniques, namely HOG features and SVM classifier. However they allowed the flexibility to use any machine learning or deep learning approach to detect vehicles. 

I have tackled this problem using a hybrid Deep Learning and Machine Learning approach. I used Tensorflow/YOLO to perform vehicle 'Detection. It is followed by a Kernelized Correlation Filters(KCF) for 'Tracking' which is a discriminative online classifier that distinguishes between a target and its surrounding environment. This approach makes it fast, able to run in realtime without a GPU! 

Original papers: [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf), [KCF]( http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf).

In my prior work, I've evaluated both HOG/SVM and YOLO for 'Detection'. I've come to the conclusion that, if you have a GPU available, then YOLO runs faster than SVM, even if you run YOLO per frame. I get decent results even if I run YOLO as little as once per second and use KCF to fill in the blanks. As an extra benefit YOLO, provides detection of 80 different classes (10 of which are of interest in automotive) with [COCO](http://cocodataset.org/) dataset. 
 

[//]: # (Image References)
[image1]: ./output_images/out2.jpg
[image2]: ./output_images/out5.jpg
[image3]: ./output_images/out8.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/yolo-pipe.png
[image6]: ./examples/grid.png
[image7]: ./examples/yolo-out.png
[image8]: https://user-images.githubusercontent.com/10645701/30238192-b8fd9a84-9574-11e7-9792-4a6529f7894c.png
[video1]: ./project_video.mp4

#### Running
Install [darkflow](https://github.com/thtrieu/darkflow) and download weights as per its instructions

    $python CombinedPipeline.py

#### Test Run

The following image is result of my Tensorflow/Yolo pipeline. Please notice it detects pedestrians, traffic lights in addition to cars. If I use different weights: VOC vs COCO, I get bus detected but lose traffic lights.

![alt text][image1]
Notice the cars detected on the opposite lane!
![alt text][image2]
![alt text][image3]

### Model Details
#### Architecture
I've used YOLOv2 implemented with TensorFlow and Keras. I've used pre-trained weights for COCO, which is a large-scale object detection, segmentation, and captioning dataset.

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

Leaky ReLU follows all convolutional layers except the last one which is "linear".

I resize the input image to 608x608 as expected by YOLO in file y2dk_wrapper.py function return_predict(). Due to skip connection + max pooling in YOLO_v2, inputs must have width and height as multiples of 32.

OpenCV loads images in the BGR format, whereas for video processing the frames are required in the RGB format. I do the conversion in the same function. The video frames are JPG and have a scale of [0, 255]. I convert it to a scale of [0, 1] before feeding it to YOLO.

The network predicts 5 bounding boxes at each cell in the output feature map. The network predicts 5 coordinates
for each bounding box, tx, ty, tw, th, and to. If the cell is offset from the top left corner of the image by (cx, cy) and the bounding box prior has width and height pw, ph, then the predictions correspond to:

    bx = σ(tx) + cx
    by = σ(ty) + cy
    bw = pwe
    tw
    bh = phe
    th
    P r(object) ∗ IOU(b, object) = σ(to)

![alt text][image6]

Wit the COCO weights, It is able to identify the 80 classes, but I restrict it to the following:

    person
    bicycle
    car
    motorbike
    bus
    truck
    traffic light

![alt text][image5]
![alt text][image7]



#### Implementation
I've used two different implementation of YOLO with TensorFlow [darkflow](https://github.com/thtrieu/darkflow) and [YAD2K](https://github.com/allanzelener/YAD2K). I've built abstraction that both can be used interchangeably. I wanted to evaluate accuracy vs speed. In case of low GPU RAM, darkflow performs faster.

Here is a rough idea of speed vs GPU requirements on NVIDIA 1080ti.[2](https://github.com/zhreshold/mxnet-yolo/issues/13)

![alt text][image8]

### Video Implementation

Here's a [link to my video result](./project_video_output.mp4)
[![See it in action on youtube](http://img.youtube.com/vi/l0_p_eeymc8/0.jpg)](https://youtu.be/l0_p_eeymc8)


#### Pipeline
The first thing I do is to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result. I've observed that undistorted images work better with YOLO and I was able to detect smaller objects, like cars on the opposite lane!

I feed the undistorted image to a Lane finder that I implemented earlier using Classic Computer Vision techniques of color transforms, and gradient thresholding. It identifies lane curvature and vehicle displacement and is robust against environmental challenges such as shadows and pavement changes.

Finally output is annotated by both the Lane Finder and Yolo tracker.

---

### Discussion

Camera distortion seems to affect YOLO detection. I was not able to detect cars on opposite lane without applying undistortion. Initially I was running YOLO every frame. It sometimes confused car with a truck. It also needed GPU to run in realtime. Then I coupled YOLO detection with KCF tracking and both problems were solved. Deep learning approach works great if you have lot of data to train it or you can reuse pre-trained network with transfer learning. It does require GPU in order to run in realtime. 

SVM/HOG also works well but the detection scope is limited to the class you trained it for. On the road, a self driving car, would encounter many unforeseen scenarios. A Deep Learning approach can cope better with wide range of objects encountered on the road, as compared to a carefully hand crafted HOG/SVM. For example if you trained your HOG/SVM on only cars, then it may fail if it encounters atypical vehicle class, like the ones used in construction and agriculture. If you want SVM to also recognize traffic signs, pedestrians, then you will need to train classifier for each individually and then combine the results. Deep learning in contrast advocates to solve the problem end-to-end. A good [writeup](https://www.analyticsvidhya.com/blog/2017/04/comparison-between-deep-learning-machine-learning/). 

SVM with sliding window does not run realtime on modern CPUs. However if you have FPGA option available on your hardware platform then acceleration is possible. Both training and acceleration of HOG/SVM requires more upfront engineering effort. 

A challenge in self driving cars is not only consistently track an object but also calculate its speed, acceleration and direction etc. I've implemented a primitive object tracker based on KCF. This is not perfect. It struggles when field of view of camera is obstructed with another moving car. A workaround can be to perform correlation between current detections and past detections to figure out which of the bounding boxes belong to which objects. It is a bit computationally involved. A better approach would be to model the object movement with Kalman Filter. A more robust solution would combine RADAR, LIDAR detection with camera to complement weaknesses of any individual sensor.


