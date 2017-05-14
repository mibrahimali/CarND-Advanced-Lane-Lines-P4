## **Advanced Lane Finding Pipeline** 

[//]: # (Image References)

[image1]: ./calibration_test/comb_calibration3.jpg "Calibration output sample 1"
[image2]: ./calibration_test/comb_calibration17.jpg "Calibration output sample 2"
[image3]: ./output_images/test1_final "Final output of Test 1 Image"
[image4]: ./output_images/test1_binary "Binary output of thresholding Test 1 Image"
[image5]: ./output_images/test1_binary_wrapped "apply prespective transformation on Binary output of thresholding Test 1 Image"
[image6]: ./output_images/test1_search_area "search area in Test 1 Image"
[image7]: ./output_images/test4_final "Final output of Test 4 Image"
[image8]: ./output_images/test6_final "Final output of Test 6 Image"
[image9]: ./output_images/straight_lines1_final "Final output of straight lines 1 Image"
[image10]: ./output_images/straight_lines2_final "Final output of straight lines 2 Image"
[image11]: ./readme_images/camera_distorsion.png "Camera Distortion Problem"
[image12]: ./readme_images/curvature.jpg "Lane Polynomial equation"
[image13]: ./output_images/project_video_out.gif "Project Video Output"
[image14]: ./output_images/challenge_video_out.gif "Challenge Video Output"
[image15]: ./readme_images/hsv.png "test 1 hsv color space"
[image16]: ./readme_images/hls.png "test 1 hls color space"
[image17]: ./readme_images/sobel_x.png "test 1 gradient in x axis"
[image18]: ./readme_images/hist.png "histogram of wrapped test 1 image"
[image19]: ./readme_images/window_search.png "window search wrapped test 1 image"

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

Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called radial distortion, and it’s the most common type of distortion.

Another type of distortion, is tangential distortion. This occurs when a camera’s lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is. This makes an image look tilted so that some objects appear farther away or closer than they actually are.

There are three coefficients needed to correct for radial distortion: k1, k2, and k3, and 2 for tangential distortion: p1, p2. In this project the camera calibration is implemented using OpenCV and a chessboard panel with 9×6 corners.

Starting with using around 20 chessboard images and for every image, we have to set for every image set of real space points and object points using "cv2.findchessboardcorners()" and calculating calibration parameters using "cv2.calibrateCamera()". we only need to do this step once. after that we use "cv2.undistort()" function to undistort camera images.

The code for this step is contained in "camera_calibration.py" and result of calibraion parameters is saved to pickle file to be used in rest of project.

here are some results after calibrating camera and undisort test images
on left Distorted images and on right undistorted ones
![alt text][image1]

![alt text][image2]

### **Color transforms**
At this point, we have an undistorted camera images. we use combination of techniques to detect potintial position of lane lines as follow:

1- transforming image to HSV color space and masking only white and yellow colors (lane lines normaly are in white and yellow) and binary thershold output.
this take place in LaneFinder class found in lines (119-140) in "advanced_lanes_finder.py" file  
![alt text][image15]

2- transforming image to HLS color space and thershold S channel.
this take place in LaneFinder class found in lines (108-117) in "advanced_lanes_finder.py" file   
![alt text][image16]

3- using Sobel Operator along the x-axis to calculate gradient of image color over RGB image space
this take place in LaneFinder class found in lines (142-162) in "advanced_lanes_finder.py" file  
![alt text][image17]

4- combining all these output in on single binary image (shown bellow sample)
this take place in LaneFinder class found in lines (371-388) in "advanced_lanes_finder.py" file  
![alt text][image4]

### **Perspective Transformation (Bird Eye View)**
Next, we use cv2.getPerspectiveTransform() and cv2.warpPerspective() functions to transform this binary image to get a bird’s-eye view image to search for lanes 
For this, we first define a set of source points and destination points. these points were hardcoded and verified by try and error methodolgy 
as follow :


| Source points | Destination points 	| 
|:-------------:|:---------------------:| 
| [250,670]    	| [250,720]   			| 
| [1050,670]   	| [1000,720] 			|
| [590,450]		| [250,0]				|
| [690,450]	    | [1000,0]   			|

We use this transformation matrix and binary image as input to cv2.warpPerspective(). We get output here a bird-eye view image of our binary image.
this take place in LaneFinder class found in lines (168-175) in "advanced_lanes_finder.py" file  

![alt text][image5]

### **Lane Pixel Detection**
Now we have a binary image of potintail area where we can find our lanes, we can search for lane pixels and then fit a polynomial of degree 2 using these pixels to get lane lines.
![alt text][image12]

for this we used window search based on Histogram of binary image as follow:

1- calculate Histogram of lower half of image
![alt text][image18]

2- using window search to cluster all possible pixels and associated with correct lane line
![alt text][image19]

this step only executed in case of first detection or we lost tracking of lane poition. but after we know potintial area where lanes are we can search for them inside a marginal space
![alt text][image6]

3- after we fit a polynomial for the current detection we check current lane base position and decide if this detection is correct or not, if correct detection we apply moving averageing filtering to smooth polynomial coeff. using last 10 times detection. 

this take place in LaneLine class found in lines (16-94) in "advanced_lanes_finder.py" file  

4- calculate radius of curvature of lane and vehicle offset w.r.t lane center position 
this take place in LaneFinder class found in lines (190-300) in "advanced_lanes_finder.py" file  

### **Final Output**
for the final output we wrap the image back to the correct image plane using the inverse prespective transformation matrix. and visualize lane area as overlay on the original image 

![alt text][image3]
![alt text][image8]

this take place in LaneFinder class found in lines (330-370) in "advanced_lanes_finder.py" file  

Final Output of Lane Finding Pipeline
---
![alt text][image13]

Potential shortcomings
---
1- this pipeline depend on many tunnable parameters to detect lanes which why it's hard to optimize.  
2- Using gradient thresholding leads to adding noise along with correct lane information 
3- pipeline utilize only color information through thersholding HSV color space searching for yello and white and HLS space for S channel. this can affect detection incase of shadows , sun glare or night condition  

Here is a look on how badly tunned parameter can affect preformance 
![alt text][image14]


Possible improvements
---
1- Enhancing this pipeline by using fusion algorithms to fuse output from multiple techniques and also track the lane information in case of non-detection scenarios.  
2- Using Deep Learning Algorithms to train a new pipeline work on all situations.   
3- Enhancing this pipeline with Shadow and sun glare detection algorithm to cancel out their effects on output
4- Use Kalmen Filter to smooth and track lane information through time 
