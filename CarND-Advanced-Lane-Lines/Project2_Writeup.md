## Project 2 Advanced Lane Finding

[//]: # (Image References)

[image1]: ./output_images/chessboard_undistort.jpg
[image2]: ./output_images/test1_undistort.jpg
[image3]: ./output_images/test1_binary.jpg
[image4]: ./output_images/straight_lines1_birdseye.jpg
[image5]: ./output_images/test3_pixels.jpg
[image6]: ./output_images/test1_final.jpg
[video1]: ./test_videos.output/project_video.mp4


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration matrix and distortion coefficients were computed using a set of chessboard images and OpenCV function cv2.calibrateCamera(). Image points were obtained using function cv2.findChessboardCorners() corresponding to 9x6 points located in the corners of the squares in the chestboard images. Corresponding objectspoints were set to (0,0,0), (1,0,0), ..., (8,5,0). An example of a distortion corrected image using function cv2.undistort() can be found below.

![Undistorted chessboard image.][image1]

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction is applied to a test image (on-road vehicle):

![Original image (left) and undistorted image (right).][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. A threshold was applied to the gradient in the x-axis and to the S-channel in an HLS color space. This step can be found in cell 3 in function pipeline() in Project2.ipynb. Here is an output of this step:

![Binary image example][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

First I compute a perspective transform matrix using a set of source and destination points and OpenCV function cv2.getPerspectiveTransform(). The points were chosen such that the transform will provide a birds eye view. The points were

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 719      | 300, 719      | 
| 595, 450      | 300, 0        |
| 685, 450      | 950, 0        |
| 1135, 719     | 950, 719      |

The perspective transform matrix was fed into OpenCV function cv2.warpPerspective() to perform the perspective transform. The code can be found in cell 4 in function pipeline() in Project2.ipynb. An example of an output can be found bellow:

![Perspective transform example][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line pixels were detected using the "Search from prior" method which means pixels searched for were inside some margin  of a polynomial curve from the previous frame. Then a new polynomial curve is computed from the obtained pixels. This is assuming the lane lines deviate little between frames. The curves are computed as a second order polynomial using numpy function polyfit() and can be found in cell 5 in function fit_poly(). The search form prior algorithm can be found in cell 6 in function search_around_poly(). The algorithm was initialized with coefficients for the left lane line and right lane line respectively

left_fit_prev = (2.13935315e-04, -3.77507980e-01,  4.76902175e+02). 
right_fit_prev = (4.17622148e-04, -4.93848953e-01,  1.11806170e+03).


Bellow is an example using a margin of +/- 100 pixels:

![Search from prior example][image5]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The lane line pixels identified in the previous step were scaled to meters, then inputted into function fit_poly() to obtain new second order coefficients A,B,C satisfying

x(y) = Ay^2 + By + C.

The radius of curvature could then be computed through

R = (1 + (2Ay + B)^2)^(3/2)/abs(2A).

Here y is chosen to be the value closest to the vehicle in meters 719 x 30/720 m.

The x-value in the center of a lane was computed as 

x_center = (x_right-x_left)/2 + x_left.

Where x_left and x_right are the x-values of the second order curves closest to the vehicle (x(y=719)).
Deviation of center in meters could then be obtained as

x_center_dev = (x_center - 1280/2)*3.7/700.

Calculation of radius of curvature can be found in cell 7 in function measure_curvature_real() and deviation of center at the bottom of function search_around_poly() in cell 6.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The main program that executes all the mentioned steps is found in cell 10 in function process_image(). Function visual_display() in cell 8 adds the radius of curvature and deviation from center as text to the image. Bellow is an output example:

![Final result][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos.output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Shadows or intensive lighting could cause pixels on the road to faultly be classified as lane line pixels and therefore compute curves that do not follow the lane lines. A solution could be to try other types of combinations of color transforms or gradient thresholding. Another is to make a sanity check to see if the curves are similar to each other in terms of  between frames.
