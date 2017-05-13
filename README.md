## **Advanced Lane Finding Pipeline** 

[//]: # (Image References)

[image1]: ./calibration_test/comb_calibration3.jpg "Calibration output sample 1"
[image2]: ./calibration_test/comb_calibration17.jpg "Calibration output sample 2"
[image3]: ./output_images/test1_final.png "Final output of Test 1 Image"
[image4]: ./output_images/test1_binary.png "Binary output of thresholding Test 1 Image"
[image5]: ./output_images/test1_binary_wrapped.png "apply prespective transformation on Binary output of thresholding Test 1 Image"
[image6]: ./output_images/test1_search_area.png "search area in Test 1 Image"
[image7]: ./output_images/test4_final.png "Final output of Test 4 Image"
[image8]: ./output_images/test6_final.png "Final output of Test 4 Image"
[image9]: ./output_images/straight_lines1_final.png "Final output of straight lines 1 Image"
[image10]: ./output_images/straight_lines2_final.png "Final output of straight lines 2 Image"


Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, The main goal is to write a software pipeline to identify the lane boundaries in a video using advanced Computer vision techniques starting from camera caliaration procedure to detect and annotate Ego lane and calculate some characteristic of these lanes (i.e. start position of lane , radius of curvature, ...etc)

![alt text][image9] ![alt text][image7]

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