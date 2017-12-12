# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataset_visualization.png "Visualization"
[image2]: ./grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./web/3.png "Traffic Sign - Speed limit (60km/h)"
[image5]: ./web/11.png "Traffic Sign - Right-of-way at the next intersection"
[image6]: ./web/12.png "Traffic Sign - Priority road"
[image7]: ./web/14.png "Traffic Sign - Stop"
[image8]: ./web/18.png "Traffic Sign - General caution"

## Rubric Points
### Consideration of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) provided as guidelines for the project.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/thorbenvh8/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I displayed an example image for each classification to get an idea what is hidden behind the classification ids and the description from signnames.csv.

Here is an exploratory visualization of the data set. It is a bar chart showing how many test data we have for each classification.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because we don't care about the color. Color doesn't have any specific function in recognizing a traffic sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so it's always between 0.1 and 0.9 instead of 0 to 255 because we want to learn from general differences in images and not some specific details.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU			|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 43       	            |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used my own MacBook Pro. I trained the model with 10 epochs, batch_size of 150, and learning rate of 0.005.

For the optimizer, first, I used softmax_cross_entropy_with_logits, then applied tf.reduce_mean() to calculate the mean of elemtent, and finally use tf.train.AdamOptimizer().


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.7%

If a well known architecture was chosen:
* What architecture was chosen?

I have chosen the [LeNet architecture](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/).

* Why did you believe it would be relevant to the traffic sign application?

"LeNet is small and easy to understand — yet large enough to provide interesting results. Furthermore, the combination of LeNet + MNIST is able to run on the CPU, making it easy for beginners to take their first step in Deep Learning and Convolutional Neural Networks."

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Starting with a 74.6% of accuracy on the first iteration and improving to 95.7% after only 10 iterations is clearly prove of the strength of this model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									|
| Priority road     			| Priority road 										|
| Stop					| Stop											|
| General caution	      		| General caution					 				|
| Speed limit (60km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For all but the last traffic sign of the speed limit the top 5 are pretty forward. For more infos please look into the notebook.

But the Speed limit traffic sign is pretty interesting.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .442         			| Speed limit (30km/h)   									|
| .291     				| Speed limit (50km/h) 										|
| .193					| Wild animals crossing 										|
| .082	      			| Speed limit (80km/h)					 				|
| .009				    | Bicycles crossing      							|

The problem is the recognizing of the letter on the traffic sign.
Having a bigger size image (more details) could resolve this problem.
