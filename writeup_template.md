# **Traffic Sign Recognition** 

[//]: # (Image References)

[sample]: ./output/sample.png "Sample"
[distribution]: ./output/distribution.png "Distribution"
[greyscale]: ./output/greyscale.png "Grayscaling"
[normalized]: ./output/normalized.png "Normalized"
[sign40]: ./output/sign40.png "Sign40"
[visual]: ./output/visual.png "Visualization"
[signs]: ./output/signs.png "Signs"

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

![alt text][greyscale]

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
* training set accuracy of 99.4%
* validation set accuracy of 94.4%
* test set accuracy of 94.1%

The LeNet architecture was choosen, that's the suggested starting point, which is from a paper discussing document 
recognition. The architecture for alphebet recognition feels like can be well reused for traffic signs recognition. 

The validation accuracy was below 90% with the original architecture. I added dropouts to each layer but the last one, 
which helped to hit 93%. 

I then adjusted the epochs from 10 to 16 to see how far away the validation accuracy can increase, then I noticed it stopped 
growing at about epoch 12, so I change to that number. I changed batch number from 128 to 32, the smaller batch number
increased validation accuracy. I also tuned the `keep_prob` from 0.5 to 0.7 which helped the validation accuracy too.

To recognize a sign we learn features from a small area to start with and build up from there, convolution layers does 
this pattern of learning well. 
 
Dropouts forces the network to learn a redundant representation of things to make sure the network doesn't rely on any
single feature. reduces overfit to the training data, 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][signs]

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
| Roundabout mandatory		| Children crossing     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 80%%. This compares lower 
to the accuracy on the test set of 94.0%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first four images, the model is relatively sure about the prediction, with > 94% probabilty each. 

But the last one's top probability is 25.3%, with the correct prediction being the second probability of 24.8%. This one 
might be hard to predict due to the three-part circle can be confused with a triangle sign. Augmenting the training data
may help with this.
  
Here I'm only showing a barchart visualization for the last sign:

![alt text][sign40]


Below are the detailed softmax probablities.

Priority road:

| Probability|Prediction | 
|:---------------------:|:---------------------------------------------:|
|0.9895009|Priority road|
|0.0103144|Roundabout mandatory|
|0.0001050|Speed limit (100km/h)|
|0.0000236|End of no passing by vehicles over 3.5 metric tons|
|0.0000195|Right-of-way at the next intersection|

Ahead only:

| Probability|Prediction | 
|:---------------------:|:---------------------------------------------:|
|1.0000000|Ahead only|
|0.0000000|No passing|
|0.0000000|Turn left ahead|
|0.0000000|No vehicles|
|0.0000000|Speed limit (60km/h)|

Speed limit (30km/h):

| Probability|Prediction | 
|:---------------------:|:---------------------------------------------:|
|0.9999628|Speed limit (30km/h)|
|0.0000231|Speed limit (50km/h)|
|0.0000142|Speed limit (20km/h)|
|0.0000000|Speed limit (70km/h)|
|0.0000000|Speed limit (80km/h)|

Pedestrians:

| Probability|Prediction | 
|:---------------------:|:---------------------------------------------:|
|0.9451081|Pedestrians|
|0.0312274|General caution|
|0.0119253|Right-of-way at the next intersection|
|0.0071728|Road narrows on the right|
|0.0026724|Double curve|

Roundabout mandatory:

| Probability|Prediction | 
|:---------------------:|:---------------------------------------------:|
|0.2534083|Children crossing|
|0.2475823|Roundabout mandatory|
|0.1855018|Right-of-way at the next intersection|
|0.0582902|Priority road|
|0.0479320|General caution| 

### (Optional) Visualizing the Neural Network.
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visualized internal state after the second convolutional layer is this:

![alt text][visual]
