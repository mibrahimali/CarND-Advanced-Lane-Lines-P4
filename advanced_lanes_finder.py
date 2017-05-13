from toolz.functoolz import excepts

__author__ = 'Mohamed Ibrahim Ali'
# Importing needed packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip
np.set_printoptions(threshold=np.inf)
from collections import deque



class LaneLine:
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients averaged over the last n iterations
        self.best_poly_fit = None
        # polynomial coefficients for the most recent fit
        self.poly_fit_history = deque(maxlen=10)
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.x = []
        # y values for detected line pixels
        self.y = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # image pixels representing lane data
        self.image_x = []
        self.image_y = []

    def update_lane(self, poly_coeff, x_pixels, y_pixels):
        if not self.detected:
            self.detected = True
            self.poly_fit_history.append(poly_coeff)
            self.best_poly_fit = poly_coeff
            self.image_x = x_pixels
            self.image_y = y_pixels
            self.calculate_lane_x_y()
            self.calculate_radius_of_curvature()
            self.calculate_lane_base_in_m()

        else:
            self.update_best_fit(poly_coeff)
            self.poly_fit_history.append(poly_coeff)
            # self.best_poly_fit = poly_coeff  # to-do update bestfit
            self.image_x = x_pixels
            self.image_y = y_pixels
            self.diffs = self.best_poly_fit - poly_coeff
            self.calculate_lane_x_y()
            self.calculate_radius_of_curvature()
            self.calculate_lane_base_in_m()

    def calculate_lane_x_y(self):
        self.y = np.linspace(0, 719, num=720)  # to cover same y-range as image
        self.x = self.best_poly_fit[0] * (self.y ** 2) + self.best_poly_fit[1] * self.y + self.best_poly_fit[2]

    def calculate_lane_base_in_m(self):
        self.line_base_pos = np.abs(self.best_poly_fit[2] - 640)*self.xm_per_pix # lane base position w.r.t vehicle center (image center)

    def calculate_radius_of_curvature(self):

        y_eval = np.max(self.y)
        # Fit new polynomials to x,y in world space
        poly_fit_world = np.polyfit(self.y * self.ym_per_pix, self.x * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2 * poly_fit_world[0] * y_eval * self.ym_per_pix + poly_fit_world[1]) ** 2) ** 1.5) / np.absolute(
            2 * poly_fit_world[0])

        # Now our radius of curvature is in meters
        # print('line_base_pos',self.line_base_pos, 'm' , 'radius_of_curvature',self.radius_of_curvature, 'm')

    def update_best_fit(self, current_fit):
        # check lane base shift value
        if np.abs(current_fit[2]-self.best_poly_fit[2]) <= 70:
            # apply averaging filter over n time stamp to smooth output
            sum = np.sum(self.poly_fit_history,axis=0)
            self.best_poly_fit = sum/len(self.poly_fit_history)
            return True
        else:
            # new detection is not correct
            print("detection is not correct")
            self.detected = False
            return False




class LaneFinder:

    perspective_transform = None
    inverse_perspective_transform = None

    def __init__(self):
        self.left_lane = LaneLine()
        self.right_lane = LaneLine()

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
        # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
        masked_gray_img = cv2.bitwise_and(img, img, mask=binary_output)
        binary_img = (cv2.threshold(cv2.cvtColor(masked_gray_img, cv2.COLOR_RGB2GRAY), 150, 255, cv2.THRESH_BINARY)[1]/255).astype('uint8')
        # Return the result
        return binary_img

    def abs_sobel_thersholding(self, img, orient='x', thresh_min=0, thresh_max=255):

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(cl1, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(cl1, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    def combine_or(self, image_1, image_2):
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

    def search_for_lane_candidates(self, img):
        if not self.left_lane.detected and not self.right_lane.detected :
            # Take a histogram of the bottom half of the image
            histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((img, img, img)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            # midpoint = np.int(histogram.shape[0] / 2)
            # leftx_base = np.argmax(histogram[:midpoint])
            # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            left_search_margin = np.int(0.4 * histogram.shape[0])
            right_search_margin = np.int(0.6 * histogram.shape[0])
            leftx_base = np.argmax(histogram[:left_search_margin])
            rightx_base = np.argmax(histogram[right_search_margin:]) + right_search_margin
            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(img.shape[0] / nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = img.nonzero()
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
                win_y_low = img.shape[0] - (window + 1) * window_height
                win_y_high = img.shape[0] - window * window_height
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
            # try:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_lane.update_lane(left_fit, leftx, lefty)
            # except:
            #     print("unable to fit left lane")
            # try:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_lane.update_lane(right_fit, rightx, righty)
            # except:
            #     print("unable to fit right lane")
        else:
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 50
            left_fit = self.left_lane.best_poly_fit
            right_fit = self.right_lane.best_poly_fit
            left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
            right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)
                self.left_lane.update_lane(left_fit, leftx, lefty)
                self.right_lane.update_lane(right_fit, rightx, righty)
            except TypeError:
                print("Unable to fit polynomial")


    def visualize_search_area(self, wraped_image):
        ploty = np.linspace(0, wraped_image.shape[0] - 1, wraped_image.shape[0])
        left_fitx = self.left_lane.best_poly_fit[0] * ploty ** 2 + self.left_lane.best_poly_fit[1] * ploty + self.left_lane.best_poly_fit[2]
        right_fitx = self.right_lane.best_poly_fit[0] * ploty ** 2 + self.right_lane.best_poly_fit[1] * ploty + self.right_lane.best_poly_fit[2]

        margin = 100
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((wraped_image, wraped_image, wraped_image)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.left_lane.image_y, self.left_lane.image_x] = [255, 0, 0]
        out_img[self.right_lane.image_y, self.right_lane.image_x] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return result

    def visualize_lane_area(self,img,wraped_image,left_lane,right_lane):

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Left Lane Radius of Curvature = {} m'.format(int(self.left_lane.radius_of_curvature)),
                    (20, 50), font, 1, (255, 255, 255), 2)

        cv2.putText(result, 'Right Lane Radius of Curvature = {} m'.format(int(self.right_lane.radius_of_curvature)),
                    (20, 100), font, 1, (255, 255, 255), 2)

        return result

    def process(self, img):
        # applying S channel threshold in HLS color space
        sbinary = self.hls_thersholding(img, thresh=(100, 255))

        # applying S channel threshold in HLS color space
        hsvbinary = self.hsv_thersholding(img)
        # applying gradient threshold in X direction
        # grad_binary_x = self.abs_sobel_thersholding(img, orient='x', thresh_min=30, thresh_max=100)

        # combiend_binary = self.combine_or(sbinary,grad_binary_x)
        combiend_binary = self.combine_or(sbinary, hsvbinary)
        combiend_binary_img = np.dstack((combiend_binary, combiend_binary, combiend_binary)) *255
        combiend_binary_wraped = self.prespective_transform_image(combiend_binary)
        combiend_binary_wraped_img = np.dstack((combiend_binary_wraped, combiend_binary_wraped, combiend_binary_wraped)) * 255
        self.search_for_lane_candidates(combiend_binary_wraped)
        search_area = self.visualize_search_area(combiend_binary_wraped)
        result_lane_area = self.visualize_lane_area(img,combiend_binary_wraped,self.left_lane.best_poly_fit,self.right_lane.best_poly_fit)
        return result_lane_area, search_area, combiend_binary_wraped_img, combiend_binary_img

if __name__ == "__main__":
    import glob
    import os

    # reading in an image
    images = glob.glob('test_images/*.jpg')
    # load saved camera calibration parameters
    camera_calibration_data = pickle.load(open("camera_calibration_parameter.p", "rb"))
    # print(camera_calibration_data)

    """
    # this part used to precess single test images to insure performance of pipeline introduced here
    for imagefile in images:
        # print(imagefile)
        file_name = os.path.splitext(os.path.basename(imagefile))[0]
        image = mpimg.imread(imagefile)
        # undistort camera images using calibration prameters obtain in previous step
        undistored_image = cv2.undistort(image, camera_calibration_data['mtx'], camera_calibration_data['dist'], None,
                                     camera_calibration_data['mtx'])

        result_lane_area, search_area_img, combiend_binary_wraped_img, combiend_binary_img = advance_lane_finder.process(undistored_image)

        # plt.figure()
        # plt.imshow(result_lane_area)
        mpimg.imsave('output_images/{}_final'.format(file_name), result_lane_area)

        # plt.figure()
        # plt.imshow(search_area_img)
        mpimg.imsave('output_images/{}_search_area'.format(file_name), search_area_img)

        # plt.figure()
        # plt.imshow(combiend_binary_wraped_img,cmap='gray')
        mpimg.imsave('output_images/{}_binary_wrapped'.format(file_name), combiend_binary_wraped_img)

        # plt.figure()
        # plt.imshow(combiend_binary_img,cmap='gray')
        mpimg.imsave('output_images/{}_binary'.format(file_name), combiend_binary_img)
    # plt.show()
    """

    def video_process(image):
        undistored_image = cv2.undistort(image, camera_calibration_data['mtx'], camera_calibration_data['dist'], None,
                                         camera_calibration_data['mtx'])
        result_lane_area, search_area_img, combiend_binary_wraped_img, combiend_binary_img = advance_lane_finder.process(undistored_image)
        return result_lane_area

    # create instance of Lane Finder class
    advance_lane_finder = LaneFinder()
    challenge_output = 'project_video_out.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    challenge_clip = clip2.fl_image(video_process)
    challenge_clip.write_videofile(challenge_output, audio=False)

    advance_lane_finder = LaneFinder()
    challenge_output = 'challenge_video_out.mp4'
    clip2 = VideoFileClip('challenge_video.mp4')
    challenge_clip = clip2.fl_image(video_process)
    challenge_clip.write_videofile(challenge_output, audio=False)



    advance_lane_finder = LaneFinder()
    challenge_output = 'harder_challenge_video_out_grad_x.mp4'
    clip2 = VideoFileClip('harder_challenge_video.mp4')
    challenge_clip = clip2.fl_image(video_process)
    challenge_clip.write_videofile(challenge_output, audio=False)