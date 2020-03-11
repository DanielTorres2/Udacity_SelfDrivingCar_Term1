# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/fig_left.jpg
[image2]: ./output_images/fig_center.jpg
[image3]: ./output_images/fig_right.jpg
[image4]: ./output_images/fig_normal.jpg
[image5]: ./output_images/fig_augmented.jpg

## Rubric Points

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
./drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model included five convolution layers and was based on Nvidia's paper provided in lesson 13. From start to end, the convolution layers had 24/36/48/64/64 amounts of filters, 5x5, 5x5, 5x5, 3x3, 3x3 window sizes, and 2x2 strides for the first three layers. Each convolution layer is followed by a RELU activation layer to introduce non-linearity to the model. The input images are normalized according to: pixel/127.5 - 1.

#### 2. Attempts to reduce overfitting in the model

Two dropout layers were introduced to fully connected layers with 100 and 50 units to prevent overfitting. Additional data was added by including images from left and right cameras as well as corresponding augmentations.

#### 3. Model parameter tuning

Training and validation sets were distributed into 80 % and 20 % respectievly of the data in driving_log.csv. The model used an adam optimizer. The probability to drop activations were set to 0.5 for both dropout layers. A batch size of 32 and number of epochs set to 5 during batch training sessions. Images were also cropped 70 pixels from the top of the image and 20 pixels from the bottom to not include irrelevant features.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle on the road. The vehicle ran three laps on track 1, driving at the center of the road, recovering from the left and right sides, and driving smoothly at curves. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the LeNet network developed from project 3. It has shown to be compatible with input data and is a convolutional netowrk I'm familiar with, why it could be a proper starting point. The model was trained using only images from center placed camera. Running a simulation caused the vehicle to drive of the road and was not sufficient to pass. The model had a relatively high mean squared error on the validation set compared to the training set which points to overfitting. To combat this, additional images from the right and left cameras were added to the data set, with corresponding augmentations, cropping images to avoid irrelevant fearues, and drop out layers for regularization. Even though the gap in mean squared error was reduced between the training and validation sets, the vehicle still left the road in simulation.

I then decided to raise the complexity of the model by using the design provided from Nvidia's paper "End to End Learning for Self-Driving Cars". The design has proven to be working on real self-driving cars and was therefore a potential candidate. Training the model using the same steps as before; include images from left and right cameras, augmentation, cropping and dropouts. The vehicle could then be obsereved to stay on the road for a complete lap witch was sufficient to pass simulation criteria.  

#### 2. Final Model Architecture

The final model architecture:

| Layer					|	Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 160x320x3 RGB image							|
| Lambda				| Input/127.5 - 1								| 
| Convolution 5x5		| 24 filters, 2x2 stride						|
| RELU					|												|
| Convolution 5x5		| 36 filters, 2x2 stride						|
| RELU					|												|
| Convolution 5x5		| 48 filters, 2x2 stride						|
| RELU					|												|
| Convolution 3x3		| 64 filters									|
| RELU					|												|
| Convolution 3x3		| 64 filters									|
| RELU					|												|
| Flatten				| 												|
| Fully connected		| 100 units										|
| Fully connected		| 50 units										|
| Fully connected		| 10 units										|
| Fully connected		| 1 units										|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded three laps. Images from all three cameras were included to provide more data. Here are three images capturing the view from left, center and right camera.

![Image taken from left side camera][image1]
![Image taken from center camera][image2]
![Image taken from right side camera][image3]

The data from the left and right cameras had their values adjusted by adding/subtracting an angle value of 0.2 due to the placement of the cameras.
I then augmented these images by flipping them horizontally and multiplying their corresponding angle values by -1, which left me with twice the amount of data. Here is an example of an augmented image.

![Image before augmentation][image4]
![Image after augmentation][image5]

I was now left with 74 916 samples. I then shuffled the data and used a batch size of 32, an adam optimizer, mean squared error as a loss function and number of epochs set to 5. 80 % of the data was used for training and 20 % for validation to observe potential over or under fitting.