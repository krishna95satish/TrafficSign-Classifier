# **Traffic Sign Recognition** 
​
## Writeup
​
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
​
---
​
**Build a Traffic Sign Recognition Project**
​
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
​
​
[//]: # (Image References)
​
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
​
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
​
---
### Writeup / README
​
### Data Set Summary & Exploration
​
#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
​
I used the pandas library to calculate summary statistics of the given data set
​
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
​
#### 2. Include an exploratory visualization of the dataset.
​
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is organized in the datasets
​
​
![alt text](../Outputs/dataset.png)
![alt text](../Outputs/traindata.png)
![alt text](../Outputs/validdata.png)
![alt text](../Outputs/testdata.png)
​
### Design and Test a Model Architecture
​
#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
​
As a first step, I decided to convert the images to grayscale because it helps to reduce the parameters since, the RGB image is made of 3 channels , we have to use filter with 3 channels as well and hence increasing the cost and reducing the accuracy
​
Here is an example of a traffic sign image after grayscaling.
​
![alt text](../Outputs/norm_gray.png)
​
As a last step, I normalized the image data because it helps the model to converge quickly to the local minima
​
​
​
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
​
My final model consisted of the following layers:
​
| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                               | 
| Convolution 5x5 (Conv1)       | 1x1 stride, Valid padding, outputs 28x28x6    |
| Leaky RELU                    |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6              |
| Convolution 5x5 (Conv2)       | 1x1 stride, Valid padding, outputs 10x10x16   |
| Leaky RELU                    |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16               |
| Convolution 1x1 (Conv3)       | 1x1 stride, Valid padding, outputs 5x5x48     |
| Leaky RELU                    |                                               |
| Max pooling & Flatten         | 2x2 stride,  outputs 1200                 |
| Fully connected       | outputs 120                                           |
| Leaky RELU                    |                                               |
| Dropout                   |                                               |
| Fully connected       | outputs 84                                            |
| Leaky RELU                    |                                               |
| Dropout                   |                                               |
| Softmax               | outputs 43                                            |
|                       |                                               |
|                       |                                               |
 
​
​
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
​
To train the model, I used the following ,
​
(1)Optimizer: AdamOptimizer
​
(2)Batch size: 200
​
(3)Number of epochs: 100
​
(4)Lerning Rate: 0.002
​
(5)Rate of Dropout: 0.7
​
(6)Loss: Cross Entropy 
​
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
​
My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 95.1%
* test set accuracy of 93.7%
​
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 A) LeNet Architecture is used since it is the easy to implement model(Ofcourse, this the only Architecture i'm familier with :P)
* What were some problems with the initial architecture?
A) Not very accurate for a self driving car.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
​
A) i changed the input image from RGB to grayscale and normalized it before feeding it to the model,
i played around with dropouts , batch size and learning rate to get the accuracy of >=93%
* Which parameters were tuned? How were they adjusted and why?
i have tuned leaning rate from 0.01 to 0.002 because my model was always overshooting away from minima , so i decided to reduce the learning rate ,
i chose leaky relu instead of normal relu because the normal was making most of the tranined weights to zero and hence saturating the learning process , to keep the backprop going i chose leaky relu.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A) since the input we are feeding into the model(in this case images) are ordered in some way,this helps in finding patters much easier,
dropouts are used to force the artificial nuerons in owr network to learn in a different way to produce the same outcomes , this is used to reduce the overfitting of our model and help generalize .
​
​
 
​
### Test a Model on New Images
​
#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
​
Here are five German traffic signs that I found on the web:
​
​
![alt text](../Outputs/webimg.png)
​
​
​
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
​
Here are the results of the prediction:
​
| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h           | 60 km/h                                       | 
| general warning               | general warning                                       |
| take left ahead                   | take left ahead                                       |
| keep to right             | keep to right                                 |
| priority road         | priority road                                 |
| 30 km/h               | 30 km/h 
​
The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
​
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
​
below is the code for getting top 5 probabilities using tf.nn.top_k() function 
​
top_5 = tf.nn.top_k(softmax_logits, k=5)
​
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    probability_5 = sess.run(top_5, feed_dict={x: new_images, keep_prob: 1.0})
    
Result is shown following image.
​
![alt text](../Outputs/webimg_prd.png)