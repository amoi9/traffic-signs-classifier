# **Traffic Sign Recognition** 

[//]: # (Image References)

[sample]: ./output/sample.png "Sample"
[distribution]: ./output/distribution.png "Distribution"
[grayscale]: ./output/grayscale.png "Grayscaling"
[normalized]: ./output/normalized.png "Normalized"
[sign40]: ./output/sign40.png "Sign40"
[visual]: ./output/visual.png "Visualization"
[image4]: ./00027.ppm "Traffic Sign 1"
[image5]: ./06357.ppm "Traffic Sign 2"
[image6]: ./07261.ppm "Traffic Sign 3"
[image7]: ./00020.ppm "Traffic Sign 4"
[image8]: ./00029.ppm "Traffic Sign 5"

Here is a link to my [project code](https://github.com/amoi9/traffic-signs-classifier/blob/master/Traffic_Sign_Classifier.ipynb), 
and the [HTML file](https://github.com/amoi9/traffic-signs-classifier/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I loaded the pickle files for training, validation and test datasets, then I used the python and numpy libraries to calculate 
summary statistics of the data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
I first picked a random image from the training set to get a sense what the inputlooks like:

![alt text][sample]

I then used the `distplot` method from the `seaborn` library to look into the distribution of labels in the three datasets,
the result is this:

![alt text][distribution]

We can see that each dataset seems to have more samples with labels between 0~10 than the ones greater than 10. The 
distribution of the labels in each dataset are similar, especailly between the training and test datasets. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I decided to convert the images to grayscale because this removes the color information 
which isn't really a factor to our classification. Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

As a last step, I normalized the image data so that our training data has a similar range. CNNs share parameters, if the
data doesn't have a similar range sharing can't happen too easily - a weight to one part may be a lot or too small to another part.

![alt text][normalized]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout					| keep probablity 0.7 |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Dropout					| keep probablity 0.7 |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Output 120        									|
| RELU					|												|
| Dropout					| keep probablity 0.7 |
| Fully connected		| Output 84        									|
| RELU					|												|
| Dropout					| keep probablity 0.7 |
| Fully connected		| Output 84        									|
| RELU					|												|
| Dropout					| keep probablity 0.7 |
| Fully connected		| Output 43        									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, with batch size = 32, epochs = 16, and learning rate = 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.1%
* test set accuracy of 94.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
 (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	Priority road      		| 	Priority road  									| 
|Ahead only    			| Ahead only										|
| Speed limit (30km/h)				| Speed limit (30km/h)										|
| Pedestrians      		| Pedestrians					 				|
| Roundabout mandatory		| Roundabout mandatory     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably 
to the accuracy on the test set of 94.0%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first four images, the model is relatively sure about the prediction, with > 99% probabilty each, with the last 
one being 37.7%. So here I'm only showing a barchart visualization for the last sign:

![alt text][sign40]


Below are the detailed softmax probablities.

Priority road:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9940363|Priority road|
|0.0054931|Roundabout mandatory|
|0.0001249|Right-of-way at the next intersection|
|0.0000847|No passing|
|0.0000683|End of no passing|

Ahead only:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9999952|Ahead only|
|0.0000044|Go straight or right|
|0.0000004|Speed limit (60km/h)|
|0.0000001|Turn left ahead|
|0.0000000|Yield|

Speed limit (30km/h):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9999998|Speed limit (30km/h)|
|0.0000002|Speed limit (50km/h)|
|0.0000001|Speed limit (20km/h)|
|0.0000000|Speed limit (70km/h)|
|0.0000000|Speed limit (60km/h)|

Pedestrians:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9909671|Pedestrians|
|0.0044251|Children crossing|
|0.0029852|Right-of-way at the next intersection|
|0.0007451|Road narrows on the right|
|0.0005997|General caution|

Roundabout mandatory:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.3769366|Roundabout mandatory|
|0.2826695|Priority road|
|0.0591221|End of all speed and passing limits|
|0.0540391|Right-of-way at the next intersection|
|0.0459142|Speed limit (120km/h)| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visualized internal state of the second convolutional layer is this:

![alt text][visual]