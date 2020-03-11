# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to grayscale, then I applied gaussian blur followed by canny edge detection.  Afterwards, I applied a mask and hough transforormation to the images. The output of the hough transform was added together with the original image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first separating the hough lines into left and right lane lines by the sign of the slope of the hough lines. To make sure hough lines that cross left and right lane are not included, a threshold of 0.5 and -0.5 was added to the slopes. Afterwards a linear function was fitted for the separated lines using numpy functions polyfit() and poly1d().


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![Images showing the result of canny edge detection, mask application, hough transformation and hough transformation added to the original image][output.png]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the car is facing a curve where the slope of the hough line were to be interpertated as its opposite sign and be separated into its opposite lane.

Another shortcoming could be that a straight line may not be appropriate to fit a curve.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to fit a line of higher polynomial in the scenario of curves.
