# **Traffic Sign Recognition** 

---

**Traffic Sign Recognition System**

Traffic signals are way of imposing rules to ensure safe and smooth transportation system in a country. So, its imperative for a  Self-Driving Car to abide by rules by understanding/detecting and acting according to the dynamic traffic signals while the car is on the road.

The goals of this project are the following:
* Load the data set from the German Traffic Sign Dataset (provided as pickle file)
* Explore, summarize and visualize the data set
* Design, train, test and experiment on model architectures and hyperparameters
* Use the model to make predictions on new images from web
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/training_distribution.png "Training Set Distribution"
[image2]: ./writeup_images/validation_distribution.png "Validation Set Distribution"
[image3]: ./writeup_images/test_distribution.png "Testing Set Distribution"
[image4]: ./writeup_images/training_distribution_after_augmentation.png "Training Set Distribution After Augmentation"
[image5]: ./writeup_images/augmented_image.png "Sample Augmented Image"
[image6]: ./writeup_images/preprocessing.png "Preprocessed Image"
[image7]: ./writeup_images/webImages.png "Images from Web"
[image8]: ./writeup_images/softmax.png "Softmax Probabilities"
[image9]: ./writeup_images/normalized_actual.png "Normalized"
[image10]: ./writeup_images/conv1.png "Visualizing Conv1 Layer"
[image11]: ./writeup_images/conv2.png "Visualizing conv2 Layer"



---
### Concepts Used
* Deep Neural Network
* Convolutional Neural Network
* Overfitting / Underfitting of Neural Network Model
* Regularization
* Hyperparameter Tuning

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of training set after augmentation is **57028**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
As can be seen below in the histogram of training set, few traffic siginals have relatively less images.

![alt text][image1]

This exploration will help us to concentrate on generating more synthetic images for the siginals having less images relative to other. Image augmentation will improve the robustness of our model.

After augmentation of training data set

![alt_text][image4]


Below is the distribution from validation set:

![alt text][image2]


Below is the distribution from testing set:

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Preprocessing**: A two level preprocessing is done for all images:

1. Converting images to gray scale. This changes input 3 channels images to 1 channel gray scale image. On the task of classifying the traffic siginals, the goal is to interpret the content of the image and map it to some traffic siginal class. Colors here will not add value as I am trying to learn the traffic siginals through learning shapes in images thus converting to single channel. Morevover, reducing the channels will reduce data hence faster computation.
    So the equals weightage is given to each channel to convert the image from 3 to 1 channel gray scale.
2. To make optimizer job easy and for numerical stability reasons, all images are normalized with simple
  (pixel_value - 128)/128
  rule.

Here is an example of a traffic sign image stepwise preprocessing:

![alt text][image6]


Through data exploration activity, it is clear that the training data contains less number of images for few traffic signs. 
So, I decided to augment the training data by generating new images for less frequent traffic signs. Also augmenting the training dataset will help models to grow more robust.

**Methodology Used**: I picked the traffic signs having less than 1000 images for them, and then generated multiple copies of the existing images with randomly perturbing following characterstics till the number of images for each traffic sign reaches above 1000 images:

1. Scaling about X,Y axis in range (0.9,.1)
2. Translation about X,Y in range (-7,7)% ~ (-2,2)px
3. Rotation in range (-15,+15)

Python framework for Augmenting the Training Set Images : imgaug
Here is an example of an original image and an augmented image:

![alt text][image5]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

**Architecture**: Below is the architecture of Neural Network utilized for training:

Final Architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, Output:28x28x6 	|
| RELU					| Output:28x28x6								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, Output:10x10x16 	|
| RELU					| Output:10x10x16								|
| Max pooling	      	| 2x2 stride,  Output: 5x5x16   				|
| Convolution 5x5	    | 1x1 stride, Valid padding, Output:1x1x400 	|
| RELU					| Output:1x1x400								|
| Fully connect.(Concat)| Output: 800x1 From Prev.Layer and last Maxpool|	
| DropOut				|           									|
| Fully connected		| Output: 43x1  								|

Other Architecture Tried:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, Output:28x28x6 	|
| RELU					| Output:28x28x6								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, Output:10x10x16 	|
| RELU					| Output:10x10x16								|
| Max pooling	      	| 2x2 stride,  Output: 5x5x16   				|
| Fully connected		| Output: 120x1      							|
| RELU					| Output: 120x1									|
| DropOut				|           									|
| Fully connected		| Output: 84x1       							|
| RELU					| Output: 84x1									|
| DropOut				|           									|
| Fully connected		| Output: 43x1  								|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Optimizer**: ADAM Optimizer

**Hyperparameters**:
  * Learning Rate: 0.0009
  * Epochs: 60
  * Batch Size: 128
  * Dropout Keep Probability while training: 0.3

**Training Parameters**:
  * Weights Initialization: Truncated Normal (Mean = 0, Standard Deviation = 0.1) 
    

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **100%**
* validation set accuracy of **96.2%**
* test set accuracy of **94.4%**

* What was the first architecture that was tried and why was it chosen?
  
  The first architecture was Lenet5 (shown in the table above as Other Architecture) but without the dropout layers between the fully connected layers.
  
  
* What were some problems with the initial architecture?
  
  With the first architecture the model is posing the problem of overfitting on the training data set.
  As I was able to achieve, 98% accuracy on the training while the validation accuracy is stuck at 78-79%.
  
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  
  As the training and validation accuracy was diverging, it was clear that model suffers from overfitting on training set. So, to introduce regaularization I introduced couple of dropout layers between the fully connected layers. This gives bumped my validation accuracy to ~96%.
  
  The final architecture I tried from the article "Traffic Sign Recognition with Multi-Scale Convolutional Networks"- Pierre Sermanel and Yann LeCunn. With this architecture, my training accuracy went to 100%, however not much improvements were detected for validation and test set.
  
* Which parameters were tuned? How were they adjusted and why?
  
  I tunned the hyperparameters in a guided manner to achieve better results from the model. 
  Methodology: Keeping all other parameters same, increasing the parameter in incremental manner to see the improvement in the accuracy.
  Applying the above methodology individually for learning rate, epochs and batch size for the model, parameters are finalized for the model.
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  Dropout layer between the fully connected layers will work well because randomly selecting the link between the fully connected layers will make the model robust as it will not de dependent on particular nodes of the fully connected layer. and this will result in more robust and better fitted model.
  

* What architecture was chosen?

  LeNet Convolutional Neural Network architecture was finally choosen with few modifications as described in article "Traffic Sign Recognition with Multi-Scale Convolutional Networks"- Pierre Sermanel and Yann LeCunn were incorporated.

  
* Why did you believe it would be relevant to the traffic sign application?

  LeNet model performed well when trained on MNIST dataset where the goal is to classify hand-written digits images to digital digits(0-9). similarly, our goal is classify the traffic images to one of (0-42) traffic siginal category. so, almost similar problem. This instigated to apply LeNet for traffic siginal problem.
  
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  With the introduction of Dropout layers between the fully connected layers and architecture described in reference article, the problem of overfitting on training data set solved as validation accuracy bumped from 78 to 96%, which is much close to training 100% as compared to before. 
  
 Best of Iterations:
 3/3/2018: LeNet5 with parameter tuning: TrA: 96.5% VaA:78.4% TeA: 77.4%
 5/3/2018: Graying the images using sum(image/3): TrA: 99.9% VaA:96.4% TeA: 94.4%
 9/3/2018: Final Architecture: TrA: 100% VaA:96.4% TeA: 94.4%
 
PS: The results may slightly vary in the html report.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
  ![alt text][image7] 

The images from the web are high quality images. However on resizing them to 32x32x1 gray scale images the quality of the images is reduced significantly, and its tough to detect the correct shapes from these bad quality images. However, since we trained our model on bad quality images from random effects on augmentation, it may able to correctly classify the images. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild Animals Crossing | Wild Animals Crossing   						| 
| Priority Road     	| Priority Road 								|
| Beware of ice/snow    | Dangerous curve to the right					|
| Roundabout Mandatory	| Roundabout Mandatory			 				|
| Stop			        | Stop    							            |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set as the accuracy there is (~94%). With just 5 images, this seems to be very valid result.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all the images, the model is relatively sure for its prediction. The same can be seen in the table below:

| Image No.  | Prediction	                   |     Softmax Probability    				| 
|:----------:|:-------------------------------:|:------------------------------------------:| 
|    1       | Wild Animals Crossing           | 0.999   				                   	| 
|    2       | Priority Road                   | 0.937 			                         	|
|    3       | Dangerous curve to the right    | 0.608			                        	|
|    4       | Roundabout Mandatory	           | 0.999			                 			|
|    5       | Stop			                   | 0.996    						            |

The top five soft max probabilities for the web images were depicted below in the horizontal bar chart   
    
![alt text][image8] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I tried to visualize the STOP traffic sign (one of the web images) to see what convolutional layers learns. Below is the normalized imput to the network:

![alt text][image9] 

Below is the visualization after the first convolutional layer which comprise of 5 feature set 
![alt text][image10] 

So, it seems in the first convolutional layer the network learn the brighter(white part of the image) as it was able to pick all the brighter part from the image.

Below is the visualization after the second convolutional layer which comprise of 16 feature set
![alt text][image11] 

It seems the second conv layer is learning some ligh level feature, which are not very clear from the images, as the original picture quality was not good.