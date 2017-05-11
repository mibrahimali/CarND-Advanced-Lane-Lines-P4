__author__ = 'Mohamed Ibrahim Ali'
# Importing needed packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
# from moviepy.editor import VideoFileClip


class LaneFinder:

    perspective_transform = None
    inverse_perspective_transform = None

    def __init__(self):
        pass

    def hls_thersholding(self, img, thresh=(0, 255)):
        # using HLS color space to thershold only Saturation channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # select S channel
        s_channel = hls[:, :, 2]
        binary_output = np.zeros_like(s_channel)
        # apply needed threshold
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        # return result
        return binary_output

    def hsv_thersholding(self, img):
        # using HSV color space to mask white and yellow colors only in picture
        image_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        temp_image = np.copy(img) * 255

        # define range of yellow and white color in HSV
        lower_yellow = np.array([20, 60, 200])
        upper_yellow = np.array([30, 255, 255])

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        # Threshold the HSV image to get only yellow and white colors
        mask_yellow = cv2.inRange(image_HSV, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(image_HSV, lower_white, upper_white)

        # combine two colors mask
        binary_output = cv2.bitwise_or(mask_yellow, mask_white)
        # Return the result
        return binary_output

    def abs_sobel_thersholding(self, img, orient='x', thresh_min=0, thresh_max=255):

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    def combine_or(self,image_1,image_2):
        return cv2.bitwise_or(image_1, image_2)

    @classmethod
    def calculate_transformation_matrices(cls):

        src = np.float32([[250, 670], [1050, 670], [590, 450], [690, 450]])
        dst = np.float32([[250, 720], [1000, 720], [250, 0], [1000, 0]])

        # use cv2.getPerspectiveTransform() to get perspective_transform , the transform matrix
        cls.perspective_transform = cv2.getPerspectiveTransform(src, dst)
        cls.inverse_perspective_transform = cv2.getPerspectiveTransform(dst, src)

    def prespective_transform_image(self, img):

        img_size = (img.shape[1], img.shape[0])
        if self.perspective_transform is None:
            self.calculate_transformation_matrices()
        return cv2.warpPerspective(img, self.perspective_transform, img_size)

    def inverse_prespective_transform_image(self, img):

        img_size = (img.shape[1], img.shape[0])
        if self.inverse_perspective_transform is None:
            self.calculate_transformation_matrices()
        return cv2.warpPerspective(img, self.inverse_perspective_transform, img_size)

    def window_search(self,img):
        binary_warped = img
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 20
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def visualize_lane(self,img,wraped_image,left_lane,right_lane):

        # Generate x and y values for plotting
        ploty = np.linspace(0, wraped_image.shape[0] - 1, wraped_image.shape[0])
        left_fitx = left_lane[0] * ploty ** 2 + left_lane[1] * ploty + left_lane[2]
        right_fitx = right_lane[0] * ploty ** 2 + right_lane[1] * ploty + right_lane[2]

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(wraped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwraped_image = self.inverse_prespective_transform_image(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, unwraped_image, 0.3, 0)
        return result

    def process(self,img):
        # applying S channel threshold in HLS color space
        sbinary = self.hls_thersholding(img, thresh=(170, 255))

        # applying gradient threshold in X direction
        grad_binary_x = self.abs_sobel_thersholding(img, orient='x', thresh_min=30, thresh_max=100)

        combiend_binary = self.combine_or(sbinary,grad_binary_x)

        combiend_binary_wraped = self.prespective_transform_image(combiend_binary)
        left_fit, right_fit = self.window_search(combiend_binary_wraped)
        return self.visualize_lane(img,combiend_binary_wraped,left_fit,right_fit)

if __name__ == "__main__":
    import glob
    import os
    # create instance of Lane Finder class
    advance_lane_finder = LaneFinder()
    # reading in an image
    images = glob.glob('test_images/*.jpg')
    # load saved camera calibration parameters
    camera_calibration_data = pickle.load(open("camera_calibration_parameter.p", "rb"))
    print(camera_calibration_data)

    for imagefile in images:
        print(imagefile)
        file_name = os.path.splitext(os.path.basename(imagefile))[0]
        image = mpimg.imread(imagefile)
        # undistort camera images using calibration prameters obtain in previous step
        undistored_image = cv2.undistort(image, camera_calibration_data['mtx'], camera_calibration_data['dist'], None,
                                     camera_calibration_data['mtx'])

        result = advance_lane_finder.process(undistored_image)
        mpimg.imsave('output_images/{}.jpg'.format(file_name), result)
        # cv2.imwrite('output_images/{}.jpg'.format(file_name), result)
        plt.figure()
        plt.imshow(result)

    plt.show()
