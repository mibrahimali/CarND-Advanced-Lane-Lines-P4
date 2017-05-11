#importing needed packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
# from moviepy.editor import VideoFileClip



def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


#reading in an image
# image = mpimg.imread('test_images/straight_lines1.jpg')
# image = mpimg.imread('test_images/test1.jpg')
image = mpimg.imread('test_images/test2.jpg')
# image = mpimg.imread('test_images/test3.jpg')
# image = mpimg.imread('test_images/test6.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
# plt.figure()
# plt.imshow(image)
# plt.title('original image')


camera_calibration_data = pickle.load(open("camera_calibration_parameter.p", "rb"))
print(camera_calibration_data)

def process_image(image):
    # undistort camera images using calibration prameters obtain in previous step
    undist_image = cv2.undistort(image, camera_calibration_data['mtx'], camera_calibration_data['dist'], None, camera_calibration_data['mtx'])
    plt.figure()
    plt.imshow(undist_image)
    plt.title('undistorted image')

    gray_img = cv2.cvtColor(undist_image, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(gray_img)
    res = np.hstack((gray_img, equ)) #stacking images side-by-side
    # plt.figure()
    # plt.imshow(res,cmap='gray')
    # plt.title('Histograms Equalization')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_img)

    # res = np.hstack((gray_img, cl1)) #stacking images side-by-side
    # plt.figure()
    # plt.imshow(res,cmap='gray')
    # plt.title('Contrast Limited Adaptive Histogram Equalization')

    gray_img = cl1

    # using different technique to detect lane lines
    # using HSV color space to mask white and yellow colors only in picture
    image_HSV = cv2.cvtColor(undist_image, cv2.COLOR_RGB2HSV)
    temp_image = np.copy(undist_image) * 255

    # define range of yellow and white color in HSV
    lower_yellow = np.array([20, 60, 200])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Threshold the HSV image to get only yellow and white colors
    mask_yellow = cv2.inRange(image_HSV, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(image_HSV, lower_white, upper_white)

    # combine two colors mask
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    masked_gray_img = cv2.bitwise_and(gray_img,gray_img, mask= mask)
    binary_img = cv2.threshold(masked_gray_img, 200, 255, cv2.THRESH_BINARY)[1]
    # plt.figure()
    # plt.imshow(binary_img,cmap='gray')
    # plt.title('yellow-white-masked-image')
    # apply morphological operator "close"
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # plt.figure()
    # plt.imshow(closing,cmap='gray')
    #
    # plt.title('applay closing operation on yellow -white mask')

    # applying gradient threshold in X direction
    grad_binary_x = abs_sobel_thresh(undist_image, orient='x', thresh_min=30, thresh_max=100)
    # plt.figure()
    # plt.imshow(grad_binary_x,cmap='gray')
    # plt.title('thersholded x grad')
    # applying gradient threshold in Y direction
    grad_binary_y = abs_sobel_thresh(undist_image, orient='y', thresh_min=30, thresh_max=100)
    # plt.figure()
    # plt.imshow(grad_binary_y,cmap='gray')
    # plt.title('thersholded y grad')
    # applying gradient magnitude threshold
    grad_binary_mag = mag_thresh(undist_image, sobel_kernel=15, mag_thresh=(80, 150))
    plt.figure()
    plt.imshow(grad_binary_mag,cmap='gray')
    plt.title('thersholded mag grad')
    dir_binary = dir_threshold(undist_image, sobel_kernel=15, thresh=(0.7, 1.3))
    # plt.figure()
    # plt.imshow(dir_binary,cmap='gray')
    # plt.title('thersholded grad dir')

    hls_binary = hls_select(undist_image, thresh=(170, 255))
    # plt.figure()
    # plt.imshow(hls_binary,cmap='gray')
    # plt.title('thersholded s channel')


    grad_mag_hls = cv2.bitwise_or(grad_binary_x, hls_binary)
    plt.figure()
    plt.imshow(grad_mag_hls,cmap='gray')
    plt.title('combined grad mag and S channel threshold')

    kernel = np.ones((9,9),np.uint8)
    closing = cv2.morphologyEx(grad_mag_hls, cv2.MORPH_CLOSE, kernel)
    plt.figure()
    plt.imshow(closing,cmap='gray')
    plt.title('closing operation on combined grad mag and S channel threshold')


    # apply perspective transformation on region of interset

    src = np.float32([[250,670],[1050,670], [590,450],[690,450]])
    dst = np.float32([[250,720],[1000,720], [250,0],[1000,0]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist_image, M, (1280, 720))
    plt.figure()
    plt.imshow(warped)
    plt.title('perspective transformation of image to focus on lanes')

    warped_binary = cv2.warpPerspective(grad_mag_hls, M, (1280, 720))
    plt.figure()
    plt.imshow(warped_binary,cmap='gray')
    plt.title('perspective transformation of closing operation on combined grad mag and S channel threshold')


    binary_warped = warped_binary

    # Assuming you have created a warped binary image called "binary_warped"
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


    # Generate x and y values for plotting

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.figure()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)



    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (1280, 720))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    return result

process_image(image=image)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)

plt.show()
