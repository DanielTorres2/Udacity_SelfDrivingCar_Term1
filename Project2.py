#!/usr/bin/env python
# coding: utf-8

# # Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## First, I'll compute the camera calibration matrix and distorion coefficients using chessboard images and a apply a distortion correction to the same images

# In[1]:


import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'qt')


# In[2]:


# Returns camera matrix and distortion coefficients given a set of chess board images  

def camera_calibration():
    
    #Object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Calibration images
    images = glob.glob('camera_cal/calibration*.jpg')


    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

mtx, dist = camera_calibration()

#img = mpimg.imread('camera_cal/calibration1.jpg')
img = mpimg.imread('test_images/test1.jpg')

undist = cv2.undistort(img, mtx, dist, None, mtx)
fig=plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
plt.xaxis()
fig.add_subplot(1, 2, 2)
plt.imshow(undist)
#cv2.destroyAllWindows()
#fig.savefig('output_images/chessboard_undistort.jpg')
#fig.savefig('output_images/test1_undistort.jpg')


# ## Next I apply a color transform consisting of a combination of a threshhold to the gradient in the x-direction and to the s-channel 

# In[3]:


# Color transform


img = mpimg.imread('test_images/test1.jpg')

def pipeline(img, s_thresh, sx_thresh):
    
    #Obtaining s-channel of the image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = sx_thresh[0]
    thresh_max = sx_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

# Plotting thresholded images

combined_binary = pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax2.imshow(combined_binary, cmap='gray')

#f.savefig('output_images/test1_binary.jpg')


# ## Next I apply a perspective transform to obtain a "birds eye view"

# In[4]:


image = mpimg.imread('test_images/straight_lines1.jpg')

undist = cv2.undistort(image, mtx, dist, None, mtx)

result = pipeline(undist, s_thresh=(170, 255), sx_thresh=(20, 100))

def warp_image(image):

    #Source points
    src = np.float32([[200, 719],[595, 450],[685, 450],[1135, 719]])
    
    #Destination points
    dst = np.float32([[300, 719],[300, 0],[950, 0],[950, 719]])
    
    #Perspective transform matrix and its inverse
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    img_size = (image.shape[1], image.shape[0])
    
    #Perspective transform
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, M_inv

warped, M, M_inv = warp_image(undist)

fig=plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(warped)
#fig.savefig('output_images/straight_lines1_birdseye.jpg')


# ## Fitting a second order polynomial to sets of pixels in the left and right lane lines

# 

# In[5]:


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    #Second order polynomial
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty


# ## Detecting pixels of lane lines using the "Search from prior" method and computing deviation from center

# In[10]:


def search_around_poly(binary_warped, left_fit_prev, right_fit_prev):
   
    # +/- window in searching around previous polynomial
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Search area for search from prior method
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + 
                    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + 
                    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + 
                    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + 
                    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))
    
    #Pixels for left and right lane line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #Obtaining polynomial coefficients
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    all_pts = np.hstack((left_pts, right_pts))
    
    #Fill area in between polynmial curves
    cv2.fillPoly(out_img, np.int_(all_pts), [0, 255, 0])
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    left_fit_prev = left_fitx
    right_fit_prev = right_fitx
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    #Computing x-coordinates for lane center position and deviation from it 
    x_left = left_fitx[ploty==719]
    x_right = right_fitx[ploty==719]
    x_center = (x_right-x_left)/2 + x_left
    x_center_dev = (x_center - 1280/2)*3.7/700
    
    return result, leftx, rightx, lefty, righty, x_center_dev, left_fit_prev, right_fit_prev

image = mpimg.imread('test_images/test3.jpg')

undist = cv2.undistort(image, mtx, dist, None, mtx)

binary_image = pipeline(undist, s_thresh=(170, 255), sx_thresh=(20, 100))

binary_warped, M, M_inv = warp_image(binary_image)

#left_fit_prev = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#right_fit_prev = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
#result, leftx, rightx, lefty, righty, x_center_dev, left_fit_prev, right_fit_prev = search_around_poly(binary_warped, left_fit_prev, right_fit_prev)
#ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
#fig=plt.figure(figsize=(20, 10))
#plt.imshow(result)
#plt.plot(left_fit_prev, ploty, color='yellow')
#plt.plot(right_fit_prev, ploty, color='yellow')

#fig.savefig('output_images/test3_pixels.jpg')


# ## Calculating radius of curvature

# In[7]:


# Calculate the radius of curvature in meters for both lane lines
def measure_curvature_real(leftx, rightx, lefty, righty):
   
    #Pixels to meters scaling factors
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    
    #Obtain polynomial coefficients based on meter data
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    #y-value closest to the car for radius of curvature calcualtion
    y_eval = 719*ym_per_pix

    #Radius of curvature for left and right lane
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval+left_fit_cr[1])**2)**(3/2))/(2*abs(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval+right_fit_cr[1])**2)**(3/2))/(2*abs(right_fit_cr[0]))  ## Implement the calculation of the right line here

    return left_curverad, right_curverad


# ## Warping back to the original image and adding deviation from center and radius of curvature as text

# In[8]:


#Adding radius of curvature and deviation from center text

def visual_display(image, x_center_dev, poly_image, M_inv, left_curverad, right_curverad):
    
    unwarped = cv2.warpPerspective(poly_image, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    final = cv2.addWeighted(image, 1, unwarped, 0.3, 0)
    #Adding text of curvatur to the image
    cv2.putText(final, 'Left lane radius = '+ str(left_curverad) + ' (m)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),3)
    cv2.putText(final, 'Right lane radius = '+ str(right_curverad) + ' (m)', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),3)

    #Define whether vehicle is left or right of center
    if x_center_dev <0:
        cv2.putText(final, 'Vehicle is ' + str(abs(x_center_dev)) + ' (m) left of center', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),3)
    elif x_center_dev >0:
        cv2.putText(final, 'Vehicle is ' + str(abs(x_center_dev)) + ' (m) right of center', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),3)
    return final


# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# ## Main program

# In[12]:


#left_fit_prev = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#right_fit_prev = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

mtx, dist = camera_calibration()

def process_image(img):
    
    #Undistorting
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    #Obtain a binary image (Color transform)
    binary_image = pipeline(undist, s_thresh=(170, 255), sx_thresh=(20, 100))
    
    #Perspective transform
    binary_warped, M, M_inv = warp_image(binary_image)
    
    #Coefficients to initialize the "Search from prior" algorithm
    try:
        left_fit_prev
        right_fit_prev
    except NameError:
        left_fit_prev = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
        right_fit_prev = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
    
    #Identify lane lines through search from prior method and obtain deviation from center
    poly_image, leftx, rightx, lefty, righty, x_center_dev, left_fit_prev, right_fit_prev = search_around_poly(binary_warped, left_fit_prev, right_fit_prev)

    #Radius of curvature
    left_curverad, right_curverad = measure_curvature_real(leftx, rightx, lefty, righty)
    
    #Invserse perspective transform and adding text of radius of curvature and deviation from center
    final = visual_display(img, x_center_dev, poly_image, M_inv, left_curverad, right_curverad)
    
    #fig=plt.figure(figsize=(10, 10))
    #fig.add_subplot(1, 2, 1)
    #plt.imshow(image)
    #fig.add_subplot(1, 2, 2)
    #plt.imshow(final)

    return final

image = mpimg.imread('test_images/test1.jpg')

fig=plt.figure(figsize=(20, 10))
plt.imshow(process_image(image))
fig.savefig('output_images/test1_final.jpg')


# In[ ]:


white_output = 'test_videos_output/project_video.mp4'

clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
print('Finished')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[ ]:




