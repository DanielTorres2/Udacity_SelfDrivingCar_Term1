# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/traindata_visu.jpg
[image2]: ./output_images/validdata_visu.jpg
[image3]: ./output_images/testdata_visu.jpg
[image4]: ./output_images/grayscale_sign.jpg
[image5]: ./output_images/five_signs.jpg


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to determine the sizes, shapes and unique classes of the data sets.

* The size of training set is 34799 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is 32 x 32 x 3 
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is bar charts showing the distribution of traffic signs with respect to the file 'signnames.csv'. The x-axis represent the label of a class and y-axis illustrates how often that class appears in the data set.

![Visual representation of training data][image1]
![Visual representation of validation data][image2]
![Visual representation of test data][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step I  converted the images to grayscale to not include color as a factor in classifying traffic sign images. Shapes, numbers, figures etc. are the important features. Here is an example of a traffic sign image before and after grayscaling.

![Grayscale of a traffic sign][image4]

As a last step, I normalized the image data by dividing the pixel values of the grayscale image by 255. This is to scale the values between 0 and 1 to make the training of the neural network faster by providing a well-conditioned region.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					|	Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 grayscale image						| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride, valid padding, outputs 14x14x6	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride, valid pading, outputs 5x5x16		|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Drop out				|												|
| Fully connected		| outputs 84									|
| RELU					|												|
| Drop out				|												|
| Fully connected		| outputs 10									|
| Softmax				| 												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam-optimizer, batch size of 128, 30 epochs, learning rate of 0.001, weight parameter of 0.015 for L2-regularization and drop-out probability of keeping an "activation neuron" of 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of  0.992
* validation set accuracy of 0.960 
* test set accuracy of 0.931

The first architecture was based on LeNet network presented in lesson 15. It was confirmed to classify trafic sign images with a validation accuracy of 0.89 and could be used as a starting point. It was noticed that the accuracy on the training set was 0.96 which suggest the model is underfitting and could benefit from regularization. The first regularization technique applied was drop out. It was observed that drop out had a bigger impact on the validation accuracy when applied to fully conected layers rather than convolutional layers and was therefore applied to both fully connected layers. Its highest validation accuracy was found with a keeping probability of 0.5. The validation accuracy exceeded 0.93 for some sessions but was not consistent. A second regularization method, L2-regularization, was therefore applied. It could be observed that the validation accuracy was highest with a weight parameter of 0.015 and provided consistent accuracy above 0.93.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Five german traffic signs][image5]

The first image is a yield sign which is unique with its downward traingular shape which should not make it difficult to classify. The second image is a 100 km/h speed limit sign. There are several speed limit signs which means the numbers on the sign will be a determining factor. There are only two signs with three digits in the data set, 100 km/h and 120 km/h, which should give a relatively high softmax probability for these two. The third image is a stop sign which is expected due to its unique eight cornered shape. The fourth image is a slippery road sign. It has an upward triangular shape which is common amongst traffic signs and therefore the image of a tilted car above curved lines will determine the classification output. This feature is however blury, and a high softmax probability for its corresponding class is not expected. The fifth image is a no entry sign. Its horizontal line is clearly able to be distinguished and the model should be able to predict this image correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image					|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											| 
| 100 km/h				| 100 km/h 										|
| Stop					| Stop											|
| Slippery road			| Slippery road					 				|
| No entry				| No entry										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92 %.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is relatively sure that this is a yield sign (probability of 1.00), and the image does contain a yield sign. The top five soft max probabilities were

| Probability			|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| Yield											| 
| 5.62e-07				| No vehicles 									|
| 2.47e-07				| Turn left ahead								|
| 9.26e-08				| Ahead only					 				|
| 9.13-e08				| Keep right									|


For the second image, the model is relatively sure that this is a 100 km/h sign (probability of 1.00), and the image does contain a 100 km/h sign. The top five soft max probabilities were

| Probability			|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| 100 km/h										| 
| 6.77e-06				| 80 km/h 										|
| 6.25e-06				| 120 km/h										|
| 6.95e-11				| Roundabout mandatory			 				|
| 1.18e-12				| No passing for vehicles over 3.5 metric tons	|

For the third image, the model is relatively sure that this is a stop sign (probability of 1.00), and the image does contain a stop sign. The top five soft max probabilities were

| Probability			|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| Stop 											| 
| 7.67e-10				| Keep right 									|
| 5.63e-10				| No entry										|
| 2.57e-11				| Turn left ahead			 					|
| 1.27e-11				| No vehicles									|

For the fourth image, the model predicts that this is a slippery road sign (probability of 0.535), and the image does contain a slippery road sign. Its probability is not as convincing as the other images due to the blury image as discussed in the first section.
The top five soft max probabilities were

| Probability			|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| 5.35e-01				| Slippery road 								| 
| 3.64e-01				| Dangerous curve to the left 					|
| 3.76e-02				| Road work										|
| 2.76e-02				| Dangerous curve to the right			 		|
| 7.47e-03				| No passing									|

For the fifth image, the model is relatively sure that this is a no entry sign (probability of 1.00), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability			|Prediction										| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| No entry 										| 
| 5.85e-08				| Round about mandatory 						|
| 5.47e-08				| No passing									|
| 1.86e-12				| Turn right ahead			 					|
| 1.44e-12				| Go straight or right							|