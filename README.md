## **Advanced Lane Finding Pipeline** 

[//]: # (Image References)

[image1]: ./calibration_test/comb_calibration3.jpg "Calibration output sample 1"
[image2]: ./calibration_test/comb_calibration17.jpg "Calibration output sample 2"
[image3]: ./output_images/test1_final "Final output of Test 1 Image"
[image4]: ./output_images/test1_binary "Binary output of thresholding Test 1 Image"
[image5]: ./output_images/test1_binary_wrapped "apply prespective transformation on Binary output of thresholding Test 1 Image"
[image6]: ./output_images/test1_search_area "search area in Test 1 Image"
[image7]: ./output_images/test4_final "Final output of Test 4 Image"
[image8]: ./output_images/test6_final "Final output of Test 4 Image"
[image9]: ./output_images/straight_lines1_final "Final output of straight lines 1 Image"
[image10]: ./output_images/straight_lines2_final "Final output of straight lines 2 Image"
[image11]: ./readme_images/camera_distorsion.png "Camera Distortion Problem"
[image12]: ./readme_images/curvature.jpg "Lane Polynomial equation"


Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, The main goal is to write a software pipeline to identify the lane boundaries in a video using advanced Computer vision techniques starting from camera caliaration procedure to detect and annotate Ego lane and calculate some characteristic of these lanes (i.e. start position of lane , radius of curvature, ...etc)

![alt text][image9] 
![alt text][image7]

The Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



### **Camera calibration**
![alt text][image11]












Potential shortcomings
---
1- this pipeline depend on many tunnable parameters to detect lanes which why it's hard to optimize.  
2- Using gradient thresholding leads to adding noise along with correct lane information 
3- pipeline utilize only color information through thersholding HSV color space searching for yello and white and HLS space for S channel. this can affect detection incase of shadows , sun glare or night condition  


Possible improvements
---
1- Enhancing this pipeline by using fusion algorithms to fuse output from multiple techniques and also track the lane information in case of non-detection scenarios.  
2- Using Deep Learning Algorithms to train a new pipeline work on all situations.   
3- Enhancing this pipeline with Shadow and sun glare detection algorithm to cancel out their effects on output
4- Use Kalmen Filter to smooth and track lane information through time 
