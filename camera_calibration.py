import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
print("calibration_set_size",len(images))
# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("calibration_matrix",mtx)
for id, fname in enumerate(images):
    img = cv2.imread(fname)
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (9, 6), None)
    # If found, draw corners overlay
    if ret:
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('calibration_test/undist_{}.jpg'.format(file_name), dst)
    imgs_comb = np.hstack((img, dst))
    cv2.imwrite('calibration_test/comb_{}.jpg'.format(file_name), imgs_comb)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open("camera_calibration_parameter.p", "wb"))

# test saved parameters in pickle file
camera_calibration_data = pickle.load(open("camera_calibration_parameter.p", "rb"))
print(camera_calibration_data)