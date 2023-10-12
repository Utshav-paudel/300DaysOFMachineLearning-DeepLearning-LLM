![machine learning image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5b5aa0e37fc4f1f2904987b7cd6bd018398c9f16/images/ml%20and%20dl.avif)
| Books and Resources | Status of Completion |
| ----- | -----|
| 1. [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction?page=1) | ✔️ |
| 2.[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)| 🏊 |
| 3.[**Intro to DeepLearning**](https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) | ✔️ |
| 4.[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)| 🏊 |
| 5.[**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)| ✔️ |


| Projects Completed |
| ----------------- |
| 1. [**Medical Insurance Price Prediction**](https://github.com/Utshav-paudel/Medical_Insurance_cost-Predictor) |
| 2.[**Iris Flower Classification**](https://github.com/Utshav-paudel/Iris-flower-calssification-webapp) |
| 3.[**California Housing Price Prediction**](https://github.com/Utshav-paudel/California-Housing-price-prediction) |
| 4.[**Collabrative filtering: Book Recommender Webapp**](https://github.com/Utshav-paudel/Book-Recommender-webapp) |
| 5.[**CNN: Bird Species Classification**](https://github.com/Utshav-paudel/Bird-Species-Classification) |
| 6.[**CNN Transfer Learning: Messy-or-CleanRoom-Detection**](https://github.com/Utshav-paudel/Messy-or-CleanRoom-Detection/tree/Utshav-paudel) |
| 7.[**Data Augmentation**](https://github.com/Utshav-paudel/Data-Augmentation/blob/5f0215d9812f54e9fae9e64c7f2673b85a5558f8/day110%20data_augmentation.ipynb) |
| 8.[**YOLO From Scratch**](https://github.com/Utshav-paudel/YOLO-Underhood) |
| 9.[**U-NET From Scratch**](https://github.com/Utshav-paudel/U-Net-Intestine) |
# Day1 
### 1. Supervised learning
Learns from being given `right answers`.  
Supervised machine learning is based on the basis of labeled data.First the data is fed to the model with both input and output and later on test data is given to make prediction by model. 
some algorithm used in supervised learning with their uses are :  
* Regression : House price prediction.
* Classification : Breast cancer detection.  
### 2. Unsupervised learning  
Learns by finding pattern in unlabelled data.
Unsupervised learning is different from supervised learning as it is not provided with labelled data.The algorithm work by finding pattern in data.  
some algorithm used in unsupevised learning with it uses are:
* Clustering : Grouping similar data points together e.g: grouping of customer , grouping of news,DNA microarray.
* Anomlay detection: Finding unusal data points e.g: fraud detection , quality check.
* Dimensionality reduction : Compress data using feweer numbers e.g : Image processing.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day2
### Univariate Linear regression  
Univariate linear regression has one dependent variable and one independent variable. With the help of indendent variable also known as input,feature we predict the output. Firstly we provide training set to our model and later on we predict the output using training set.
### Cost function   
A cost function is a measure of how well a machine learning model performs by quantifying the difference between predicted and actual outputs.  
`lower the value of cost function better the model`  
![cost function](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/276707b43ff92bd2f822312e545e90d26c4358c9/images/day2%20costfunction.png)  
* [linear regression model](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/276707b43ff92bd2f822312e545e90d26c4358c9/code/day2%20univariate%20linear%20regression%20model.ipynb)
* [cost function](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/276707b43ff92bd2f822312e545e90d26c4358c9/code/day2%20cost%20function.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day3
### Gradinet descent
![gradient descent img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1c0e9bd63ea4a5d41d8f417b4da9650b1dc3c567/images/day3%20gradient%20descent.png)  
Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.It is made cleared in below image.  
![gradient descent equation img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1c0e9bd63ea4a5d41d8f417b4da9650b1dc3c567/images/day3%20gradinet%20descent%20equation.png)
* [gradient descent](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/41535e5806b7fc119dc833a4486e6e7d15e9bfbe/code/day5%20gradient%20descent.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day4
### Learning rate
Learning rate `alpha` in gradient descent should be optimal.
* If learning rate is too small gradient descent may be too slow and take much time.  
* If learning rate is too large gradient descent may overshoot and never reach minimum i.e fail to converge,diverge.  
![learning rate](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/2fd63f02907afd95dfec81f13eb576065a5a0f29/images/day4%20learning%20rate.png)

* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day5
### Multiple linear regression
Multiple linear regression in machine learning model that uses multiple variables called as features to predicts the output.
![multiple linear regression model](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/41535e5806b7fc119dc833a4486e6e7d15e9bfbe/images/day5%20multiple%20linear%20regression.png)
### Vectorization 
In muliple linear regression calculation is done using vectorization as it perform all calculation simultaneously and parallely and speed up the arithmetic operations.
![vectorization img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/41535e5806b7fc119dc833a4486e6e7d15e9bfbe/images/day5%20vectorization.png)
* [vectorization in numpy](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/41535e5806b7fc119dc833a4486e6e7d15e9bfbe/code/day5%20numpy%20vectorization.ipynb)
* [Multiple linear regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4291831060fcce086ac8677076c92803d42d9fb6/code/day6%20multiple%20linear%20regression.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day6
### Feature scaling 
When you data features has very large range,too small range gradient descent may take large time so data is rescaled to normal similar range called feature scaling.
some popular feature scaling techniques are:  
* mean normalization
* Z score normalization  
![feature scaling type](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4291831060fcce086ac8677076c92803d42d9fb6/images/day6%20feature%20scaling%20techniques.png)   
Feature scaling visual representation  
![feature scaling representation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4291831060fcce086ac8677076c92803d42d9fb6/images/day6%20feature%20scaling%20representation.png)
### Choosing correct learning rate 
First we make sure gradient descent is decreasing over the iteration by looking at learning curve if it is working properly we choose correct learning rate by starting with smaller learning rate and increase it gradually.
* [feature scaling  and learning rate](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a5fad158c61e2bc72d1d088909a9ba4aafd4e92a/code/day6%20feature%20scaling%20and%20choosing%20learning%20rate.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day7
### Feature engineering 
Feature engineering means designing newfeatures by transforming or combining original features which maybe very important in prediciting the output.  
for e.g: we have to predict the price of swimming pool and we have length breadth and height of swimming pool as features now we can used feature engineering to create our new feature which is volume which is very important in predicting the price of swimming pool.
### Polynomial regression
Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:  
y= b0+b1x1+ b2x12+ b2x13+...... bnx1n   
It is used incase of non linear dataset.  
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/cbcb932cb36976f30699482499555b839ce77a42/images/day7%20polynomial%20regression.png)
* [featured engineering and polynomial regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e6e8b83dc852a9551557585ee93b65c48832e2df/code/day7%20feature%20engineering%20and%20polynomial%20regression.ipynb)
* [Linear regression using scikit learn](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e6e8b83dc852a9551557585ee93b65c48832e2df/code/day7%20linear%20regression%20using%20scikit%20learn.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day8
### Classification
Classification is a type of supervised learning in machine learning, where the goal is to predict the class label of an input data point.For example, we may want to classify emails as spam or not spam, or classify images as cats or dogs.
### Logistic regression  
Logistic regression is a type of algorithm used for classification problems. It works by estimating the probability of an input data point belonging to a particular class. For example, it may estimate the probability that an email is spam or not spam, or the probability that an image is a cat or a dog.

To estimate these probabilities, logistic regression uses a mathematical function called the logistic function, which maps the input data to the probability space. The logistic regression algorithm then learns the relationships between the input features and the target class by adjusting weights, or coefficients, assigned to each input feature. These weights are adjusted to maximize the probability of the correct classification.

In the end, logistic regression outputs the predicted class for each input data point, based on the estimated probabilities. This can be useful for a wide range of classification tasks, from predicting diseases to detecting fraud.
![logistic reg img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/317f551e71cd3e2deb546e7f37c578639d3c6f27/images/day8%20logistic%20regression.png)
* [classification](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/317f551e71cd3e2deb546e7f37c578639d3c6f27/code/day8%20classification.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day9
### Sigmoid function
The sigmoid function is a mathematical function that maps any input value to a value between 0 and 1. It is commonly used in logistic regression to model the probability of a binary outcome. The sigmoid function has an S-shaped curve and is defined as follows:

σ(z) = 1 / (1 + e^(-z))

where z is the input value to the function. The output of the sigmoid function, σ(z), is a value between 0 and 1, with a midpoint at z=0.

The sigmoid function has several important properties that make it useful in logistic regression. First, it is always positive and ranges between 0 and 1, which makes it suitable for modeling probabilities. Second, it is differentiable, which means that it can be used in optimization algorithms such as gradient descent. Finally, it has a simple derivative that can be expressed in terms of the function itself:

d/dz σ(z) = σ(z) * (1 - σ(z))

This derivative is used in logistic regression to update the model coefficients during the optimization process. 
* [Sigmoid function](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/8030ab207dc51472a4ad8cc13fd211587132a4ee/code/day8%20sigmoid%20function.ipynb)
### Decision boundary
The decision boundary is the line that separates the area where y=0 and where y=1.It is create by our hypothesis function.
In logistic regression, the decision boundary is the line (or hyperplane in higher dimensions) that separates the different classes of the target variable. The decision boundary is determined by the logistic regression model, which uses the input variables to predict the probability of belonging to a certain class.  
![decision boundary image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/55bb303deeace7b4800319a38312bfe0202fc59d/images/day9%20decision%20boundary.png)

* [decision boundary](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/8030ab207dc51472a4ad8cc13fd211587132a4ee/code/day8%20decision%20boundary.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)\
# Day10
### Gradient descent in Logistic regression  
Logistic Regression Ŷi is a nonlinear function(Ŷ=1​/1+ e-z), if we put this in the above MSE equation it will give a non-convex function as shown:  
![loss function image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a3e7cb1c6715f32f4f2744083b8eb6f69374f56c/images/day10%20cost%20function%20in%20logistic%20regression.jpg)
* When we try to optimize values using gradient descent it will create complications to find global minima.

* Another reason is in classification problems, we have target values like 0/1, So (Ŷ-Y)2 will always be in between 0-1 which can make it very difficult to keep track of the errors and it is difficult to store high precision floating numbers.

The cost function used in Logistic Regression is Log Loss.  
![log loss image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a3e7cb1c6715f32f4f2744083b8eb6f69374f56c/images/day10%20logloss.png)  
Cost function for logistic regression  
![cost function image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c5b6cff9ce3fbe68c4fa6864ecb8787a9eb48516/images/day11cost%20function%20in%20logistic%20regression.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day11
### Gradient Descent in logistic regression  
Gradient Descent in Logistic Regression is an iterative optimisation algorithm used to find the local minimum of a function. It works by tweaking parameters w and b iteratively to minimize a cost function by taking steps proportional to the negative of the gradient at the current point.  
Gradient descent in logistic regression looks similar to gradient descent in linear regression but it has different value for function.  
![gradient descent in logistic regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/15bcacd6e5d8f971d54629bbf414fe63aa17b0df/images/day11%20gradient%20descent%20for%20logistic%20regression.png)
* [gradient descent in logistic regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/aecd0846a7e0f80efa23131e0eed695715df4c09/code/day11%20gradient%20descent.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day12
### Underfitting 
It is a situtation when the training set doesnot fit well. It happen when data has high bias.  
### Overfitting  
It is a situation when the training set fit extremely well . It is also known as data with high variance.
### Addressing overfitting 
* Collecting more training example
* Select features include/exclude
* Reduce the size of parameters i.e "Regularization".
* [overfitting](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/82251c369534b9dd1fa500b5e7c39ac8baeb0015/code/day12%20overfitting%20example.ipynb)
### Regularization  
Regularization is a technique to reduce the parameter and prevent overfitting of data. It has a term called lambda whose value if larger result underfitting and smaller result overfitting it also called penalty term.  
![regularization term](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/367b379e09f12e6493fc91b24b7c617399eaacf4/images/day12%20regularization%20image.png)
* [Regularization in linear regression and logistic regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/82bf8ece95114cc3d2b597749553a888514942da/code/day13%20regularization%20in%20linear%20regression%20and%20logistic%20regression.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day13
### Neural network
Neural network is an computer algorithms that try to mimic the brain.neural network is made of a input layer that take input data and hidden layer does all the computation and output layer displays the output.  
![neural network image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/86901d1f2b3b8017f98025ec7f4310ff38d602aa/images/day13%20neural%20network.svg)  
Why neural network ?  
Neural network is necessary because it increase performance of machine learning algorithm compared to traditional algorithm like linear regression and logistic regression because it uses multiple and more algorithm in a neural network to make better prediction and performances.  
![why neural network](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/86901d1f2b3b8017f98025ec7f4310ff38d602aa/images/day14%20why%20neural%20network.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day14
### Neural network notation  
In neural network.
* neuron is represneted by subscript.
* neural network layer is represented by superscript.
### Forward propagation in neural network  
Forward propagation refers to storage and calculation of input data which is fed in forward direction through the network to generate an output. Hidden layers in neural network accepts the data from the input layer, process it on the basis of activation function and pass it to the output layer or the successive layers. Data flows in forward direction so as to avoid circular shape flow of data which will not generate an output. The network configuration that helps in forward propagation is known as feed-forward network.  
* [neuron and layer](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/f355d3a41ea555a8183799abadabff766fab088f/code/day14%20neuron%20and%20layer.ipynb)

* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day15
### Neural network implementation in tensorflow
Neural network can be easily implemented in tensorflow as below:  
* [neural network implementation in tensorflow](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/af55e5c1ed24a4083b95a66e16d91e0c52984f69/code/day15%20coffee%20roasting%20in%20tensorflow.ipynb)
![neural network implementation in tf](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/be3f04d372d0afa4f03c25c18cf1a3df2c75d8fa/images/day15%20neural%20network%20implementation%20using%20tensorflow.png)
### AGI  
AI is mainly classified into two type: ANI and AGI
* AGI:An AGI is a hypothetical intelligent agent that can learn to accomplish any intellectual task that human beings or other animals can perform. It is defined as an autonomous system that surpasses human capabilities in the majority of economically valuable tasks

* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day16
### Vectorization in neural network
In neural network vectorization helps to perform calculation simultaneously and save a lot of time. It can implemented as :  
![vectorization in neural network](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a05af1a6599eb33e7aea0dc22857abb590961275/images/day16%20vectorization%20for%20neural%20network.png)
### Neural network implementation in code
![neurall network representation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a05af1a6599eb33e7aea0dc22857abb590961275/images/day16%20neural%20network%20representation.png)  
![code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a05af1a6599eb33e7aea0dc22857abb590961275/images/day16%20neural%20network%20code%20for%20above%20representation.png)
* [Neural network to identify handwritten binary](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/13c7fdd68ca420478aedbffee1d1f3ee2dba9bdf/code/day16%20handwritten%20digit%20recognition%20neural%20network.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day17
### Model Training steps  
Model training is simplified in 3 steps :  
1. Specify how to compute output given input x and parameters w,b (define model)  
  * linear regression (y = ax + b) 
  * logistic regression (y = 1/(1 + np.expt(-z)))  
2. Specify loss and cost  
  * Mean square error 
  * Logistic loss (BinaryCrossentropy) (loss = -y*np.log(f_x) - (1-y)*np.log(1-f_x)  
  Note: Cost is the sum of loss for all training examples.  
3. Train on data to minimize cost(Gradient descent)  
  w = w - alpha*dj_w  
  b = b - alpha*dj_b   
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
### Activation function
There are different activation function for different purpose some of the most commonly used are :  
* Linear acitvation function(activation='linear')  
This is used for regression problem where result is negative/positive.
* ReLU (activation = 'ReLU')  
This is used for regression problem where result should be positive always and it is faster as compared to sigmoid function.
* Sigmoid function(activation='sigmoid')   
It is used for classification problems where result must be on/off and it is slowere as compared to ReLU.  
NOTE:`For hidden layer we choose ReLU as activation and for output layer we choose activation according to our problems,because if we choose sigmoid in hidden layer than neural network becomes very slow so it better to choose Relu in hidden layer`  
![activation function](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/87630d03f23299852372e56f49b5c4f5be8789c9/images/dasy17%20activationn%20function.webp)
* [ReLU implementaion](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/87630d03f23299852372e56f49b5c4f5be8789c9/code/day17%20ReLU%20activation.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day18
### Multiclass classification 
Target y can take on more than two possible values. In this case of multiclass classification we use Softmax regression.
### Softmax regression 
Softmax regression is the generalization of logistic regression for multiple classs.   
Its output is calculated as:  
![Softmax regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a8bb8ecf939c1d1b76dc31d62b8367f34ab7e437/images/day18%20softmax%20regression%20.png)

### Cost for softmax regression
Cost for softmax regression is also known as cross-entropy loss. It is obtained as.
![Cost for softmax regression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a8bb8ecf939c1d1b76dc31d62b8367f34ab7e437/images/day18%20Crossentropy.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day19
### Improved Implementation of softmax/logistic regression in neural network
Our normal implementation of softmax cause some of numerical roundoff error so for the more numerical accurate implementation of softmax regression we use linear activation in output layer and passing `from_logits = True as parameter in loss at model.compile()`.  
You can get more insight by looking at image below:  
![Numerical accurate implementation of softmax](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/78b44e1cbd3aec966e8ec070377f761fc5f80d40/images/day19%20imporved%20implementation%20of%20softmax.png)
* It can be implemented similarly for softmax neural network.
* [Numerically accurate implementation vs normal implementation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/78b44e1cbd3aec966e8ec070377f761fc5f80d40/code/day19%20softmax%20prefered%20way%20vs%20not%20preferred%20way.ipynb)
### MultiLabel classification
Multilabel classification is a type of classification problem in machine learning where each instance can be assigned to multiple classes or labels simultaneously. In other words, instead of predicting a single class for an instance, the goal is to predict a set of labels that are applicable to that instance.   
Here is difference between multiclass and multilable classfication 
![multi label classification](Multilabel classification is a type of classification problem in machine learning where each instance can be assigned to multiple classes or labels simultaneously. In other words, instead of predicting a single class for an instance, the goal is to predict a set of labels that are applicable to that instance.)
* [Multi class classification](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/78b44e1cbd3aec966e8ec070377f761fc5f80d40/code/day%2019%20multiclass.ipynb)
### Advanced optimization
Adam algorithm is used for advanced optimization in neural network.
* If learning rate is smaller adam algorithm increases it automatically.
* If learning rate is larger then adam algorithm decreases it automatically.
Note : It stands for Adaptive Moment estimation.
It is used as:  
```python
model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = tf.kearas.losses.SparseCategoricalCrossentropy(from_logits=True))
```

### Additional Layer types:
Some of layer types of neural network are : 
* Dense Layer (Fully Connected Layer): A dense layer is a basic layer where each neuron is connected to every neuron in the previous layer. It is characterized by its weight matrix, bias vector, and activation function. Dense layers are commonly used in feedforward neural networks and can learn complex patterns and relationships in the data.

* Convolutional Layer: Convolutional layers are primarily used in convolutional neural networks (CNNs) for analyzing grid-like data, such as images. These layers perform convolutions, applying filters to the input data, and capturing local patterns and features. Convolutional layers are effective in image recognition, object detection, and other computer vision tasks.   
Convaulational neural network are faster in computation and need less training data as compared to Dense Layer.  
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day20
### Back Propagation  
Backpropagation, or backward propagation of errors, is an algorithm used in machine learning to adjust the parameters of a neural network by calculating the gradients of a loss function with respect to the network's weights and biases. It propagates the error from the output layer to the input layer, allowing the network to learn and improve its predictions.
* [Backward propagation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/32905b341938026ab9db9af034a4e664f664e5c5/code/day20%20backward%20propagation.ipynb)
### Debugging a learing algorithm
When we have large error in prediction we can debugg or learning algorithm as follow:  
* Get more training examples.
* Try smaller set of features.
* Try getting additional features.
* Try adding polynomial features.
* Try decreasing/increasing lamda regularizing parameter
### Evaluating a choosen model: 
You can evaluate a model by splitting data into trian/test and calculating cost for both training set and test set .
### Model selection:
The most effective way of model selection is by
* splitting data into train/cross validation /test set
* Calculating error for cross validation set and selecting model with less cross validation error .
* Calculation error for test data of model with less cross validation error.
* [Model selection using train/cv/test](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/fae56941f9d55feccb10a03ef3fdce4625b56863/code/day20%20train_crossvalidation_test%20split%20for%20model%20selection.ipynb)
### Machine learning diagnostic
A test that you can run to gain insight into what is/isn't working with a learning algorithm to gain guidance into improving its performance .
Ml model can be diagonse by looking at bias and variance:   
When model has high bias and variance it is not doing well.   
![how to diagnose high bias and variance](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/346f73b367af2df6c84ca81a038b66d6ad7d22b0/images/day21%20diagonising%20high%20bias%20and%20variance.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day21
### Bias/Variance
* High bias: When model has large difference between baseline performance and Training error then it is called high bias and it also indicates underfitting.
* High variance: When model has large difference between training error and cross validation error then it is called high variance and it also indicates overfitting.
* High bias and variance: When model has large difference between training error , cross validation error and baseline performance then it is called both high bias and high variance.  
![image show bias and variance](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/346f73b367af2df6c84ca81a038b66d6ad7d22b0/images/day21%20bias%20and%20variance.png)
### Choosing regularization parameter
To choose good regularization paramter.
* First,Apply all regularization value and get different cross validation error and the smallest cross validation error indicated a good regularization term.
* NOTE : Right model neither has high variance and neither has high bias
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day22
### Diagonising bias and variance
#### If your algorithm has high bias:
* Try getting additional features.
* Try increase polynomial degree.
* Try decreasing regularization term.
#### If your algorithm has high variance:
* Getting more training examples.
* Trying decreasing set of features.
* Try increasing regularization term.
### Bias and Variance in neural network:
* If your neural network has high bias try increasing size of neural network.
* If your neural network has high variance try increasing training sets.
Note : Most probably larger neural network perform well as long as appropriate regularization term is choosen.
![bias and varince in neural net](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/123ae0650d91421c315b670aed72f4c7ec9c659b/images/day22%20bias%20and%20variance%20in%20neural%20network.png)
* [Diagonising bias and variance](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/123ae0650d91421c315b670aed72f4c7ec9c659b/code/Day22%20diagonising%20bias%20and%20variance.ipynb)
*  📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day23
### Iterative loop of ML Development
Ml development revolve around following steps:
* Choosing architecture(model,data,etc)
* Training model
* Diagnostics(bias,variance and error analysis)
![iterative loop img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/da0508acb47b6af85c845293f74e0dbc47b51ae7/images/day23%20iterative%20loop%20of%20%20ml%20development.png)
### Error analysis
It is the process to isolate,observe and diagnose erroneous ML predictions to understand pockets of high and low performance to the model.
### Adding more data
Adding more data is mostly useful to make better predictions and data can be added by following ways:  
1. Data augmentation: Modifying an existing training example to create new training example. e.g: Data augmentation by adding distortion.  
2. Data synthesis: Using artifical data inputs to create a new training example. It is mostly used for computer vision applications.
### Transfer learning 
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.  
It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.
* Download neural network parameters pretrained on large dataset with same input type (e.g: images,audio,text) as your application (or train your own).
* Further train(fine tune) the network on your own data.
*  📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day24
### Full cycle of machine learning project 
Machine learning project is iterative process which is as below:  
![ml lifecycle](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfe65718b2b6c3b14e6fcf8c1e654b64afd4a713/images/day24.png)
### Deployment 
Mlops focuses on making ml model to be used in largescale and deployment is basically done by:
![ml deployment](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfe65718b2b6c3b14e6fcf8c1e654b64afd4a713/images/day24%20deployment%20of%20ml%20model.png)

### ethics,bias and faireness of machine learning
While developing machine learning application we have to take care of biasness and negative case like :  
1.Deepfake  
2.Genrating fake content for commercial and political purposes  
3.Ml model biasing in loan provider,job selection.

### Precision/recall
* precision : It tell of all positive prediction how many are actually positve.
* recall : It tell of all real positive cases how many are actually predicted positive.
![precsion recall img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfe65718b2b6c3b14e6fcf8c1e654b64afd4a713/images/day24%20precision%20and%20recall.png)
### Trading off precsion and recall 
* When threshold is higher, precision become higher and recall lower down
* When threshold is lower, precision become lower and recall become higher.
NOTE : To select better precsion and recall value we use `F1 score` which is the harmonic mean of precision and recall.  
* [Summary of Advance learning algorithm](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfe65718b2b6c3b14e6fcf8c1e654b64afd4a713/code/day24%20summary%20of%20advanced%20learning%20algorithm.ipynb)
*  📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day25
### Decision Tree
A decision tree is a type of supervised machine learning used to categorize or make predictions based on how a previous set of questions were answered. The model is a form of supervised learning, meaning that the model is trained and tested on a set of data that contains the desired categorization. 
![decision tree image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c47f3b6ac073db8e2d634ef749042dca8ad9eab2/images/day25%20decision%20tree.png)
### Decision tree learning 
1. How to choose what feature to split at each node ?  
* Maximize purity
2. When to stop splitting 
* When a node is 100% one class
* when splitting of node will exceed a maximum depth
* when imporovements in purity score are below a threshold
* when no. of examples in a node is below a threshold.
*  📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day26
### Measuring purity
In decision tree entropy is the measure of level of impurity and helps to find purity of classes. lower impurity means higher purity.  
![entropy](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e3a817ab172d6abf9c552365c2842ed09621a314/images/day26%20entropy%20as%20impurity.png)
### Information gain
We can calculate the information gain by subtracting the weighted average entropy of the resulting subsets from the entropy of the original node. The formula for information gain is:  
Information Gain = Entropy(node) - Σ((subset_size/total_size) * Entropy(subset))  
* By maximizing information gain, decision trees aim to find the attribute that provides the most useful and informative splits, leading to more accurate classification.
### Decision tree learning
* Start with all examples at the root node
* Calculate information gain for all possible features, and pick the one with the highest information gain
* Split dataset according to selected feature, and create left and right branches of the tree
Keep repeating splitting process until stopping criteria is met:
1. When a node is 100% one class  
2. 2. When splitting a node will result in the tree exceeding a maximum depth  
3. 3. Information gain from additional splits is less than threshold When number of examples in a node is below a threshold  
![dc tree](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e3a817ab172d6abf9c552365c2842ed09621a314/images/day26%20decision%20tree%20splitting.png)
# Recursive splitting
Recursive splitting refers to the iterative process in decision tree construction where a dataset is divided into smaller subsets based on specific conditions. It involves recursively selecting attributes to split on and creating branches that further partition the data until a stopping criterion is met, resulting in a tree-like structure.
*  📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day27
### One hot encoding
If a categorical features can take on k values, create k binary features(0 or 1 values) is call one hot encoding.
### Spitting for continuous variable
For continuous variable we have to choose threshold with higher information gain and split on the basis of that threshold.
### Regression tree
It is a decision based tree used to predict continous variables.
* for selecting best split for regression tree we find variance reduction and the biggest variance reduction is considerd to be the best split.
* Note: Variance reduction = variance of root node - average weighted variance of leaf node
* [Regression tree](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/61a61fb8b16300dbe9cd84a5196a99f0def98ce3/code/day27%20decision%20tree.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day28
### Tree ensemble 
Single decision tree is very sensitive to data so the process of combining many decision tree to build more robust system is called tree ensemble.
the prediciton of tree ensemble is obtained by majority result of tree.
### Random forest algorithm
A random forest algorithm is a machine learning technique that combines the predictions of multiple decision trees to make more accurate and robust predictions. It works by creating an ensemble of decision trees, where each tree is trained on a random subset of the data and uses a random subset of features. The final prediction is then made by averaging or voting the predictions of all the trees in the forest. The random forest algorithm is effective at handling complex datasets, handling missing values, and avoiding overfitting.
* Random forest has two term that explain this algorithm they are Bootstrapping and aggregation and the combination of this is called Bagging
* Boostrap : The selection of subset of training example (sampling with replacement) where the training example can be repeatded is called bootstrap
* Aggregation : The selection of majority of result from ensembles tree is called aggregation.
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e9f5ce59b23328fc72b9d06af34e1ba5c1ed6db3/images/day28%20random%20forest.png)
### XG Boost
In XG boost we basically pick the training examples that were misclassified previously instead of training all samples.
It is implemented as :   
![xg boost img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e2d84dfea2fc053bd1c6de84520adbc9713c4a93/images/day28%20Xg%20boost.png)
### When to use decision tree 
* Decision tree works pretty well on structured data but is not recommended for unstructured data like audio,video and images 
* It is faster compared to neural network
* Small decision trees may be human interpretable.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day29
### Unsupervised learning
Machine leanring algorithm that find patterns on unlabelled data.
### K-means Clustering Algorithm
K-means clustering is an unsupervised machine learning algorithm used for partitioning a dataset into K distinct non-overlapping clusters. Each data point in the dataset is assigned to the cluster with the nearest mean (centroid). The algorithm aims to minimize the within-cluster variance, also known as the "inertia."

Here's a step-by-step overview of the k-means clustering algorithm:

* Initialization: Randomly select K data points from the dataset as the initial cluster centroids.

* Assignment: Assign each data point to the nearest centroid. This is done by calculating the Euclidean distance (or other distance metrics) between each data point and each centroid, and assigning the data point to the cluster with the closest centroid.

* Update: Recalculate the centroids of each cluster by taking the mean of all the data points assigned to that cluster.

* Repeat: Repeat steps 2 and 3 until convergence or a maximum number of iterations is reached. Convergence occurs when the centroids no longer move significantly between iterations or when the algorithm reaches the predefined maximum number of iterations.

* Final Clusters: Once convergence is achieved, the algorithm outputs the final cluster assignments, where each data point belongs to one of the K clusters.  
![k means cluster](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0b42f7fba067ff92974b209532654e381edc8996/images/day29%20k%20means%20clustering.png)
### Cost function for k-means clustering(distortion)
The cost function for k-means clustering is commonly referred to as the "inertia" or "within-cluster sum of squares." It measures the sum of squared distances between each data point and its assigned centroid within each cluster. The goal of k-means clustering is to minimize this cost function.

Mathematically, the cost function (J) for k-means clustering can be defined as:

J = Σᵢ Σⱼ ||xᵢ - μⱼ||²  
* [K-means clustering sample](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4957fca69c2b4e65b1964620f40eacf8895fb293/code/day29%20k%20means%20clustering.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day30
### Anamoly detection
Anomaly Detection is the technique of identifying rare events or observations which can raise suspicions by being statistically different from the rest of the observations. Such “anomalous” behaviour typically translates to some kind of a problem like a credit card fraud, failing machine in a server, a cyber attack, etc.
* To apply anamoly detection we use gassuian distribution/normal distribution
### Anamoly detection vs Supervised learning
### Anamoly detection
We will use anamoly detection when there are very small number of example that are positive(anamoly) and large number of negative example .Since small number of positive examples it is hard to learn from training examples.e.g: fraud detection
### Supervised learning
We will use supervised learning when large number of positive and negative examples are present. Since there are enough positive examples to train model and predict supervised learning is effective. e.g Email spam detection
* NOTE: Before using anamoly detection we have to make sure our data is normal.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day31
### Recommender system
Recommender systems are a type of machine learning algorithm used to suggest items to users based on their preferences or behavior. These systems are widely used in various applications like e-commerce, movie streaming platforms, music apps, and more.
### 1.Content based Recommendation
In content-based recommender systems, the algorithm recommends items based on the similarity between the content/features of the items and the user's preferences. The similarity is typically computed using techniques such as cosine similarity or Euclidean distance. Here's an overview of the mathematical steps involved:

* a. Feature Representation: Each item and user is represented as a feature vector. Let's denote an item's feature vector as x and a user's preference vector as p. These vectors consist of numerical values that represent the attributes or characteristics of items or users.

* b. Similarity Measure: The similarity between two feature vectors, x and p, can be computed using cosine similarity. The cosine similarity between x and p is defined as:

similarity(x, p) = (x · p) / (||x|| * ||p||)

where (x · p) represents the dot product of vectors x and p, and ||x|| and ||p|| denote their respective Euclidean norms.

* c. Recommendation: To recommend items to a user, the system calculates the similarity between the user's preference vector and the feature vectors of all items. The system then ranks the items based on their similarity scores and recommends the top-rated or most similar items.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
### Collaborative Filtering Recommender Systems:
Collaborative filtering recommender systems make recommendations based on the preferences or behavior of other similar users or items. Let's explore the two main approaches: user-based and item-based collaborative filtering.

* a. User-based Collaborative Filtering:

In user-based collaborative filtering, the algorithm finds similar users based on their past interactions or ratings and recommends items that the similar users have liked. Here are the mathematical steps involved:

i. User Similarity: The similarity between two users, u and v, can be computed using techniques such as cosine similarity or Pearson correlation. The similarity score measures the likeness of their past interactions.

ii. Prediction: To predict a user's preference for a particular item, the system combines the ratings of similar users. The predicted rating, denoted as r_hat(u, i), for user u and item i is calculated as a weighted average of the ratings of similar users:
```python
    r_hat(u, i) = ∑ (sim(u, v) * r(v, i)) / ∑ |sim(u, v)|
#where sim(u, v) represents the similarity between users u and v, r(v, i) denotes the rating of user v for item i, and the summation is performed # over all similar users v.
```
where sim(u, v) represents the similarity between users u and v, r(v, i) denotes the rating of user v for item i, and the summation is performed over all similar users v.
iii. Recommendation: The system recommends items with the highest predicted ratings to the active user.
*  Item-based Collaborative Filtering:

In item-based collaborative filtering, the algorithm identifies similar items based on the past interactions or ratings of users. It then recommends items that are similar to the ones the user has already liked. Here's a summary of the mathematical steps involved:

i. Item Similarity: The similarity between two items, i and j, can be computed using techniques such as cosine similarity or Pearson correlation. The similarity score measures the likeness of user preferences for the items.

ii. Prediction: To predict a user's preference for a particular item, the system considers the user's past ratings for similar items. The predicted rating, denoted as r_hat(u, i), for user u and item i is calculated as a weighted average of the user's ratings for similar items:
```python
  r_hat(u, i) = ∑ (sim(i, j) * r(u, j)) / ∑ |sim(i, j)|

  #where sim(i, j) represents the similarity between items i and j, r(u,

```
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day32
### Normalization
Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
### Limitation of collabrative filtering
How to 
* rank new items that few user have rated ?
* show something reasonable to new users who have rated few items ?
How to use side information about items or users:
* Item: Genre, movei stars, studion...
* User: Demorgraphics(age,gender,location), epressed prefernces,..
Note: This limitation of collabrative filtering can be addressed by content based filetring
### [Tensorflow implementation of collabrative filtreing](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/e34e047cf5b50496b478695c5aad219690b69b18/code/day32%20collabrative%20filetering%20for%20movie%20recommendation.ipynb)
### Content base recommendation
It uses both content and user data and using neural network create vector for content and vector for user and its dot product give prediction.
### Content base recommendation for large items.
When our website or app has large number of content to recommend like thousands and millon of item it is carried out in two steps:
#### 1.Retrival
From large number of content retrival is carried out for selective content for further ranking. for e.g:  
For movies recommendation:  
1.for 10 movies watched by user retrieve similar movies.  
2.for most viewed 3 genres find top 10 movies.  
3.find top 20 movies in country.  
At last combined retrived item in list and remove duplicated and items already purchased.   
#### 2.Ranking
Apply model to retrived data to find suitable item and display ranked item to user.   
Note: Retriving more items result in better recommendation but takes more time to analyse try it offline and find suitable number of retrival for better and relevant recommendations.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
### Day33
#### [Implementation of content based filering](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/cb9ea35528675120c2c2f1f41bd837002fd5d515/code/day33%20content%20based%20filtering%20for%20movie%20recommendation.ipynb)
### Dimensionality reduction
Dimensionality reduction is a technique used to reduce the number of features in a dataset while retaining as much of the important information as possible. In other words, it is a process of transforming high-dimensional data into a lower-dimensional space that still preserves the essence of the original data
* Feature selection : Feature selection involves selecting a subset of the original features that are most relevant to the problem at hand. The goal is to reduce the dimensionality of the dataset while retaining the most important features. There are several methods for feature selection, including filter methods, wrapper methods, and embedded methods. Filter methods rank the features based on their relevance to the target variable, wrapper methods use the model performance as the criteria for selecting features, and embedded methods combine feature selection with the model training process.

* Feature Extraction: Feature extraction involves creating new features by combining or transforming the original features. The goal is to create a set of features that captures the essence of the original data in a lower-dimensional space. There are several methods for feature extraction, including principal component analysis (PCA), linear discriminant analysis (LDA), and t-distributed stochastic neighbor embedding (t-SNE). PCA is a popular technique that projects the original features onto a lower-dimensional space while preserving as much of the variance as possible.
### Principal Component analysis(PCA)
Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.
*  It works on the condition that while the data in a higher dimensional space is mapped to data in a lower dimension space, the variance of the data in the lower dimensional space should be maximum.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day34
### Step by step calculation of PCA
PCA is used to reduce higher dimension data to lower dimension without losing it essence. PCA can be calculated in following steps:
* Mean centring data
* Finding covariance matrix 
* Finding eigen value and eigen vector 
* Eigen vector with largest eigen value has highest variance and is ready for selction.
* [PCA](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/19d213dde7bdb7b78360ae2e2688be0499efc076/code/day34%20PCA.ipynb)  
After applying PCA to handwritten digit having 784 features we got optimal solution of pca at around 250 that explains nearly to 90% of variance.   
![optimal solution of PCA](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/19d213dde7bdb7b78360ae2e2688be0499efc076/images/day34%20%20optimal%20features%20of%20pca.png)  
### 2D plot of 784 feautres

|![2d plot of 784 features ](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/19d213dde7bdb7b78360ae2e2688be0499efc076/images/day34%202d%20pca%20plot.png) 
### 3D plot of 784 features
![3d plot of 784 features](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/19d213dde7bdb7b78360ae2e2688be0499efc076/images/day34%203d%20pca%20plot.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day35
### Reinforcement Learning
Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones. In general, a reinforcement learning agent is able to perceive and interpret its environment, take actions and learn through trial and error. 
It reward is calculated as:  
![reward png](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/84b741cc0d1d1529090a83d92dc4bf50e8c32cf5/images/day35%20reward%20formula)
### Markov Decision Process(MDP)
It state that future depends on current state .  
![MDP](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/84b741cc0d1d1529090a83d92dc4bf50e8c32cf5/images/day35%20reinforcement%20learning.png)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day36
### State action value function
In state action value function, represented by Q(s,a)   
Q(s,a) = Return , If you 
* start in state s
* take action a(once)
* Then behave optimally after that
Note: Behaving optimally means taking action which bring maximum Q(s,a).  
Implementaion of state action value function:
![State action value function img](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/ab4e0367f6e3b922fa20a01c614a11ded789f6d3/images/day36%20state%20action%20function.png)
### Bellman equation
Bellman equation explain the return in two step first one is immediate reward and second one is reward from behaving optimally starting from state s.  
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/ab4e0367f6e3b922fa20a01c614a11ded789f6d3/images/day36%20bellman%20equation.png)
### Random stochastic environment 
Due randomness and uncertainity in enviroment it becomes diffcult for reinforcement learning so to avoid this we caluclate Expected return(i.e average return) in placce of return only .   
It is calculated as : Q*(s, a) = E[R(s, a, s') + γ ∑ P(s'|s, a) max(Q*(s', a'))],   
* where E represent average or Expected return .
* [State action value implementation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/ab4e0367f6e3b922fa20a01c614a11ded789f6d3/code/day36%20state%20action%20value%20function.ipynb)
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day37
### Discrete state and continous state
* Discrete state : It is a state in reinforcement learning where the number of possible state are distinct and countable. Discrete state spaces are common in many RL problems, such as board games, puzzle-solving, and decision-making tasks with a finite number of states.
* Continuous state : It is a state in reinforcement learning where the number of possible state are in continous range . Continuous state spaces are encountered in various RL applications, including robotics, control systems, and real-world scenarios where states are represented by continuous measurements.
### Differences in Discrete and continous state
Discrete state spaces can often be represented using tabular methods, where the agent maintains a value function or a Q-table to learn and update action values for each state. On the other hand, dealing with continuous state spaces often requires `function approximation` techniques, such as using `neural networks, to approximate the value function or policy`. Continuous state spaces also pose challenges for exploration strategies, as the agent needs to explore a potentially infinite space effectively.
### Exploitation(greedy) VS Exploration
It also know as epsilorn greedy policy
* with probability 0.95, pick the action a that maximizes Q(s,a) - Greedy (Exploitation)
* with porbability 0.05, pick action a randomly. (Exploration)
* This means epsilon = 0.05
* Start with high epsilon and decrease gradually
### Refinement of reinforcement learning by minibatches and softupdate
### Mini batches
When we have large number of training examples our iterative process like gradient descent and other iterative process on reinforcement learning like trainin neural network becomes slower so we divide main training examples to differen subsets called mini batches
### Soft Update
Soft update in reinforcement learning refers to a technique used to refine the learning algorithm by updating the parameters of a target network gradually. This process involves interpolating between the parameters of the target network and the parameters of the online network.
* 📚Resources  
course:[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?page=1)
# Day38
### Building of Books recommender system using Collabrative filtering.
Collaborative filtering recommender systems make recommendations based on the preferences or behavior of other similar users or items.
In this book recommendation system I calculated similarity scores between two users to find the euclidean distance and recommendation was made on the basis on two nearer items and most of the collaborative recommender system work like this. 
#### To avoid cold start in collabrative filtering:
* The movie with more the 50 rating was included.
* The user who have rated more than 200 books was included.
Here is snippet of the project hope you get some insight riding this   
![book recommender system](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/d4e06acc0a1f9da8332d9f3c9769f8a2b1fd1151/images/day38%20movie%20recommender%20system.png)
# Day39
### Califorina Housing Price prediction
Started my project on California housing price and cleared my concepts on data pipeling, Batch learning and Online learning, perfomed the fetching and loading of data with EDA to gain some insight on data.  
**NOTE**
* **Batch learning vs Online learning** : Batch learning is commonly used when the dataset can fit into memory and when the computational resources are sufficient to process the entire dataset at once. It is often used in offline scenarios where there is no need for real-time or incremental learning.  
Online learning is particularly useful when dealing with streaming data or when the dataset is too large to fit into memory. It allows for real-time learning and adaptation as new data arrives. Online learning algorithms often have lower memory requirements and can adapt to concept drift, which is the phenomenon where the underlying data distribution changes over time.
* **Cost for linear regression** : Most of the time cost for linear regression is calculated by Root Mean Square Error(RMSE) but when data has lots of outlier we use Mean Absolute Error (MAE)
![calforinaday1](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/24ea2817f4f7092bfa8ff472f0a2b8000dd8183d/images/day39%20Calfornia-dataloading%20and%20Eda.png)
![Eda result](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5bac9c7462071160bf06432031e4bc21bdf2d304/images/day40%20California%20eda%20resul.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day40
### California Housing Price Prediction continued..
* Today I Created test data , and splitted data on the basis of train-test-split and also with stratifcation split to remove imbalance in data and create same proportion.   
**Creating test data**
![testdata](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5bac9c7462071160bf06432031e4bc21bdf2d304/images/day40%20california%20creating%20test%20set.png)
**Creating Training and test data with random sampling and stratification sampling**
![stratification](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5bac9c7462071160bf06432031e4bc21bdf2d304/images/day40%20traintest%20vs%20strarification%20split.png)
### Different methods of sampling :
1.Random sampling 
2.Systematic sampling
3.Stratified sampling
* Stratified sampling: Stratified sampling can be useful in train-test split when dealing with imbalanced datasets. In this case, the dataset may have significantly different proportions of classes or subgroups. By using stratified sampling, we can ensure that the training and test sets maintain the same distribution of classes or subgroups as the original dataset. This helps to prevent bias and provides a more accurate evaluation of the model's performance.  
![sampling in train test split](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/9d17986685775af19e615468114ec3f0089fcac4/images/day40%20stratification%20to%20solve%20imbalance%20in%20class.png)
# Day41
### California Housing Price Prediction
* **Visualization with geographical data** : I plotted geographical visual with respect to population density and housing price to gain better understanding of data   
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5955221f5ea3581accba7a4207088bed30cd68c9/images/day41%20california%20visualization%20of%20geographical%20data%20indication%20housing%20price%20and%20popn.png)
* **Correlation**: Also plotted scatterplot for coefficient of correlation of median_housing price with respect to different features and found out that it has high correlation with median_income but there was some straight line forming in middle of data which need to be filtered before training for better performance.  
![correlation fig](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5955221f5ea3581accba7a4207088bed30cd68c9/images/day41%20california%20correlation%20of%20median_house_price%20with%20features.png)
Here is code hope you gain some insight from it :   
![visualization](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5955221f5ea3581accba7a4207088bed30cd68c9/images/day41%20california%20housing%20code.png)
![correlation code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/06802a4eb2c9cd3d53efbc6443c0ad1133d8852a/images/day41%20california%20checking%20for%20correlation.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day42
### California Housing Price Prediction
### Expermenting with attribute combinations 
* I created some new combination of feature like room per house , bedroom ration, population per house and found that room per house has done well then other features, it got some high negative correlation that indicated the less bedroom ratio more the price.
* Also Prepared data for machine learning algorithm by separating the features and target, and perform cleaning of data, replced missing values by filling it with median as it is less destructive.
![code for data cleaning](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1a4f6b1318fd7e320d8673a0cd35d2e4e16149df/images/day42%20californina%20data%20cleaning.png)
### Use of Simple Imputer
* The benefit of using SimplImputer is that it will store the median value of each feature: this will make it possible to impute missing values not only on the training set, but also on the validation set,the test set, and any new data fed to the model.
![photo of simple imputer use](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1a4f6b1318fd7e320d8673a0cd35d2e4e16149df/images/day42%20california%20handling%20missing%20using%20Imputer.png)
### Handling of text and categorical data
* Text and categorical data can be handled by using ordinal encoder and One Hot encoding but incase of ordinal encoder it think data nearby data are more similar than far data which is not the case in Oceanproximity so we use onehot encoding.
![handling text and categorical data](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1a4f6b1318fd7e320d8673a0cd35d2e4e16149df/images/day42%20handling%20text%20and%20categorical%20data.png)

* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day43
### Feature Scaling
* Feature scaling is one of the most important transformation you need to apply to  your data. Without feautre scaling most model will bias one feature with another. Two ways of feature scaling are : 1.Min-max scaling  2.Standarization.
* `Never use fit() or fit_transform() for anything else than training set.`
![feature scaling image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/92006ed56d2928f2f3dd6faa3a5121d612bc2657/images/day43%20featurescaling%20california%20code.png)
### Bucketing/Binning
The transformation of numeric features into categorical features, using a set of thresholds, is called bucketing (or binning)
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a57d93feaa03386c050725ae72b64cf9b36a848b/images/day43%20bucketing%20concept.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day44
### Column Transformer
* Column transformer is a versatile tool in machine learning that allows for the application of different preprocessing steps to different columns or subsets of columns in a dataset. It simplifies the preprocessing workflow, enhances reproducibility, and improves the efficiency of feature engineering in machine learning tasks.
### Pipeline
* Pipeline refers to a sequence of data processing steps that are applied in a specific order. It combines multiple operations, such as data preprocessing, feature engineering, and model training, into a single cohesive workflow. It make easier to apply same preprocessing to training and test set
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/07d1d3751f7f719a5ecdef3583ed8f435332ea30/images/day44%20columntransformer%20and%20pipeline%20california%20housing.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day45
### Select and train a model
* I trained some model like LinearRegression, DecisionTreeRegressor and RandomForestRegressor and found out RMSE very high in LinearRegrssion which indicated underfitting and RMSE 0 in DecisionTreeRegressor which was heavily overfitting and RMSE was comparatively low on RandomForestRegressor.So, I  find that RandomForestRegressor can be a good choice
![selecting model](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/640b2d5857962c1dcde47ca02e92316c9f989e5a/images/day45%20california%20selcting%20and%20training%20model.png)
### Evaluation of CrossValidation and Fine tunig the model
* Performing CrossValidation also showed that Random forest was good choice despite of some overfitting and After some tuning in RandomForestRegressor using GridSearch CV I got some good hyperparameter and model perform more better than before and RMSE was also reduced.
![last day code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/640b2d5857962c1dcde47ca02e92316c9f989e5a/images/day45%20evaluation%20%20of%20cv%20and%20finetuning%20of%20model.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day46 
### Classification 
* Today I dive deep into classification and revised some topic using MNIST like binary classifier, measuring accuracy using cross validation, confusion matirces, precision and recall, precision recall tradeoff.
* Measuring accuracy is more complex in classification than in regression some of the methods of measuring classification accuracy are : Using cross validation, confusion matirces, precision and recall, ROC curve.
The below code show implementation of measuring accuracy in classification using different method in MNIST dataset hope you gain some insight:  
![da46_measuring_accuracy](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/f4a114bc917e059316e68b44b4bbbb0164c0f9b4/images/day46%20accuracy-measure%20in%20classifcation.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day47
* Today I revised my classification concept on Multiclass Classification, error analysis, Multilabel Classification, Multioutput Classification.
* Multiclass classifier are capable of handling more than two class. Some of the algorithm that does multiclass classification are: LogisticRegression, RandomForestClassifier and GaussianNB. Also can be done using multiple binary classifier like SGDClassifier and SVC.    
1.One Vs All (OVA/OVR): In One Vs All classification there are N Classifier model for N classes. Model with highest score is selected for particular class calssification.  
2.One Vs One (OVO): In One Vs One classification N classes have N*(N-1)/2 classifier model. In this, we have one classifier model for each class against every other class.   
* Also performed error analysis using confusion matrix
* Implemented Multilabel classification : A classification system that
outputs multiple binary tags is called a multilabel classification system. 
* Multioutput Classification : It is a classification that has multilabel and each label can have multiple class.
Below is the implementation of what I have learned today.
![classification](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/cf150b177fc5a0fe63c1076e2f8f14005d2baeb1/images/day47%20classification%20part2%20complete.png)
[Classification implemetation code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/3e9027fd4ecb963912e9abcf29ae75487f8b4d06/code/Day47%20Classification%20day2.ipynb)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day48
* Today I dived deep into training model and what happen underhood the  sickit learn model with help of simplest Linear regression model and learned about minimizing model cost function using gradient descent.
* Gradient descent is used to find best parameter for reducing cost function. tips:`while using gradient descent all features most have same scale e.g use standardscaler or it will took very long time to converge`. Batch gradient descent takes whole batch of training data at each steps so it is terribly slow during large datasets.
* To solve the problem of Batch gradient descent we use stochastic gradient descent as named suggest step of this gd is random and stochastic gradient descent picks a
random instance in the training set at every step and computes the gradients based on it rather than taking whole data. But since it is random the steps may never settled to optimal minimum so this can be solved by decreasing learning rate gradually know as simulated annealing. `To use stochastic gradient descent with linear regression we can use SGDRegressor`.
* Mini-batch GD com‐
putes the gradients on small random sets of instances called mini-batches. The main
advantage of mini-batch GD over stochastic GD is that you can get a performance
boost from hardware optimization of matrix operations, especially when using GPU
![gd code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/96eea78783eef9e0b3377b4906700bbded88082e/images/day48%20training%20model.png)
[Implementation of code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/ad991fc189ea0625418925e8ec5285d3157e4461/code/day48%20training%20models.ipynb)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day49
* Today I revised my concept on polynomial regression, learning curve, and about the condition of underfitting, overfitting and it solution.
* Polynomial regression model is used for non-linear data, we used cross-validation to see how our model is performing to check whether it is underfitting or overfitting, it can be also checked in polynomial regression with the help of Learning curve. If the model is underfetting it is better to add some new features or to use different model but adding more data doesnot help in underfitting and IF the model is overfitted it can be solve by regularization.
![polynomial](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/813584a9c1f941ba8385955813f6df6cf755ee8c/images/day49_polynomial_regression.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day50
* Today I learned about Ridgeregression also known as L2 regularization used to solve overfitting problem it can be aslo used with SGDRegressor directly by putting l2 in penalty and don't forget to divide alpha with m.
  ![ridgregression](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/065aa698ce47fe3f7eeea8b8f14537c25b7778e8/images/day50%20ridgeregression.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day51
* Today I learned about Lasso Regression also known as L1 regularization and elastic net regression. Lasso Regression is used in case of high dimensional data and it can perform dimensionality reduction by setting coefficient zero of less important feature on increasing regularization paramter lambda which cannot be done by ridge regression.
* Elastic Net regression : This regularization term is the middle ground of both Ridg and Lasso it is calculated as weighted sum of both ridge and lasso regularization term, elastic net is preffered when more features have strong correlation or number of features is greater than training instances.
* Early stopping : A very different way to regularize iterative learning algorithms such as gradient
descent is to stop training as soon as the validation error reaches a minimum. This is
called early stopping.

![regularization](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6ba6ae180635ead3632908210c5da2993c2b8c99/images/day51%20lasso%2Celastic%20and%20early%20stopping.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day52
* Today I revisied my concept of logistic regression and softmax regression, logistic regression is used for classification problems and has costfunction called logloss, whereas softmax regression is used in case of multiclass classification.I spend sometime using logistic regression to classify iris flower based on petal length whether it is virginica or not and observed it decision boundary and used softmax regression to do same.
You can gain some insight on implemenation of what I learned :
![classfication](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/b70296a2df927669af312a6e02d9639651286e25/images/day52%20logistic%20regression.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day53
* **Support Vector Machine:**
SVM is a popular machine learning model that does linear or non-linear classification, regressio and even novelty detection and it perform well with small and medium datasets.The core idea of SVM is :<br>
1.Start with lower dimension data<br>
  2.Move data in higher dimension<br>
  3.Find a support vector classifier that separates the higher dimension data.<br>
* Kernel function : It is  function that transform data from lower dimension to higher dimension to find support vector classifiers. some of them are : Linear kernel, polynomial kernel, Radial bias function (RBF) kernel and sigmoid function.
* Kernel trick : Calculating the high dimensional relationships without actually transforming the data to higher dimension is called kernel trick. It reduces the mathmatical computation.
 ![Implementation of SVM](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/16e98e7906623a887151e7860bf334d9b598bc01/images/day53%20svmpart1.png)
* 📚Resources
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day54
* **Polynomial Kernel :** The polynomial kernel is a kernel function that calculates the similarity between two data points in a feature space using a polynomial function. It is defined as:
K(x, y) = (α * x^T y + c)^d
![formulas](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/24759b9895374f5330ffa21be484128280046124/images/day54svm%20formula.png)
* **RBF Kernel :** The RBF kernel, also known as the Gaussian kernel, is a popular kernel function that measures the similarity between data points based on their radial distance in a feature space. It is defined as:
K(x, y) = exp(-γ * ||x - y||^2)
* Both the polynomial kernel and the RBF kernel leverage the kernel trick, which is a method used in machine learning to implicitly transform data into a higher-dimensional feature space without explicitly calculating the transformed features. The kernel trick allows algorithms to efficiently operate in this higher-dimensional space by only computing the kernel function values between data points.
![code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5fc4af583c0fbe98ff972997807347437d588f13/images/day54%20svm%20part2.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day55
* Today I continued my learning on SVM, SVM classes like LinearSVC adn SGDclassifier doesnot support kernel trick but it is supported by SVC class.SVM can also be used to solve regression problem by tweaking parameter epsilon, width of margin can be increased by increasing epsilon.
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/b23b510d141d12008e4dfd33034a1dd8d435fcb0/images/day55%20SVM%20regression.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day56
* Today I spended my time revising concept on decision tree from book and implementing it. Decision tree can perform both classification and regression task and are the fundamental unit of random forest which is one of the most important algorithm in machine learning.Decision tree uses entropy and gini impurity to look how model is doing.Gini impurity is default in decision tree classifier and best in case of large dataset as computationally it is faster than entropy while entropy provide more balanced classes.
* Regularization : Regularization can be done in decision tree by reducing freedom to decision tree and controlling following parameter like max_leaf_nodes:Maximum number of leaf nodes,
min_samples_split:
Minimum number of samples a node must have before it can be split, 
min_samples_leaf:
Minimum number of samples a leaf node must have to be created,
min_weight_fraction_leaf:Same as min_samples_leaf but expressed as a fraction of the total number of
weighted instances.
* Increasing min parameter and reducing max parameter regularize decision tree.
  ![code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfa1d535452953f3ee5548b7a9c7eea7175ec5a9/images/day56%20decision%20tree.png)
  ![graph](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dfa1d535452953f3ee5548b7a9c7eea7175ec5a9/images/day55%20decisiontree-graph.jpeg)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day57
* **Decision Tree for regression:** Decision tree for regression is also similar to classification decision tree it recursively create split until the pure class is obtained. The selection is split is done by information gain using variance reduction.
* Also learned to reduce max_depth and increase min_sample_split and min_sample_leaf to overcome overfitting in decision tree of regression.
* Sensitivity to axis rotation : Decision tree does well when split is perpendicular to axis but when split is not perpendicular to axis data may not generalize well so we use scaling and pca to transform data and genralize it.<br>
* Hyperparameter tuning in decision tree result in high variance which is solved by averaging many decision tree known as ensemling and such ensemled tree is random forest which one of the popular and widely used ml algorithm.
Implementation of them is given below hope you get some insight.
![regression decision tree](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/cad00f7108f2341b874ce271adee3c2ab6d89cb9/images/day57%20decision%20tree%20final.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day58
* **Ensemble learning**: Today I revised my concept on ensemble learning and I have to say that this is the beauty of machine learning. I enjoyed topics like Wisdom of crowd that hold major concept of ensemble learning (wisdom of crowd = If we average the decision of large crowd we get our result which is close to actual result.) and learned about voting classifier, bagging, boosting and stacking .
* **Voting Classifier**: voting classifier train different model and either get prediction based on majority vote called hard voting or get prediction based on probability of each model prediction and averaging it called soft voting. `for soft voting we have to set hyperparameter as 'soft' and for svc you have to set probability hyperparameter to True.`
  Implementation of my learning is given below hope you get some insight from it.
  ![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c0a7ccc7ada7107071978b02f2e90ed0c87bf0a3/images/day58%20ensemble%20learning%20voting%20classifier.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day59
* **Bootstrap Aggregation**: In this method of ensemble learning whole data is divided into different sample and selected without replcement known as bootstrap and different model is feded with different sample taken and the result of all model is averaged for regression and voting classifier is done for classfication known as aggreagation. And the data missed in sample is known as oob(out of bag).
  Implementation of bagging:
  ![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/bdf3498176dfd7dd4505d6dfecd199f1775e6989/images/day59%20bagging.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)  
# Day60 
* **Random patches :** Random patches means taking sample of both features and training instances.
* **Random subspaces :** Random subspaces means taking sampel of features but using all training instances.
 This techniques are used for higher dimensional data like images
![day60](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4bd0928c89792b4f024d1a4223a6fc984bd8bce8/images/day60%20random%20subspaces%20and%20random%20patches.jpg)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day61
### Random Forest
* Today I revised my concept on random forest which is generally the ensemble of decision tree based on bagging concepts. It can be used by importing RandomForestClassifier or RandomForestRegressor according to requirement.Also random forest are very good with feature importance and tell which feature has how much importance , it can be handy for feature selction and it score can be accessed by feature_importances variable.
  ![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dec7414355f595a9c8a403b456abc4c75016a0c8/images/day61%20random%20fores.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day62
### Boosting
* Boosting basically trains the predictors sequentially each trying to correct it predecessor.Some of the popular boosting method are Ada boosting and gradient boosting.
* Ada boosting : Ada boost focuses more on the training instances that the predecessor underfit sequentially and works on updated weight. If ada boositng overfit it can be regularized by reducing the number of estimator or number of boosting stages.
* Gradient boosting : It also works sequentially but instead of updating training instances weight it works by fitting the predecessor on residual error of previous model.
Implementation of boositing is shown below hope you gain some insight.
![boosting](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/885082a576488db162910a7b3c8ab1bdf58351f5/images/day62%20boositng.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
  # Day63
  ### History based gradient boosting
  * HistGradientBoosting is faster than gbrt and it has two features that it allows missing values and categorical features.
    It is implemented as :
    ![hgb](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/a73f2f95575e110c30a054996e662adacd9d2460/images/day63%20hgb.png)
  ### Stacking
  * In this ensemble methods there are different model for predicting result based on training instances and the aggregation of this model prediction is done by another final model known as meta learner or blender. The result of base predictor will be feature for blender and the target of trainig instances will be reused to blender model.
![stacking](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/7936b77b3a1f8b62995000e08d9f4523e8d085ed/images/day63%20stacking.png)
* [Implementation of ensemble learning](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/8ffacdafdca9037de748b80274292b2beaf7aba8/code/day59-62%20Ensemble%20learning.ipynb)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day64 
### Dimensionality reduction
I have already mentioned and studied about dimensionality reduction technique using pca today I am going to revised and implement it.
Dimensionality reduction is process of reducing dimension of data without losing it essence and it is done to speed up training and somtime to reduce noise. Two methods of dimensionality reduction are projection and manifold learning.
### PCA
It is the most popular technique for dimension reduction it reduce dimension by choosing hyperplane that is closest to the data by preserving variance.
You can find the number of components/features to use by setting ratio of variance to preserve ideally 95 % in parameter n_components. You can gain some insight form below code.
![pca ](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c5dafdd02713edcd0ad81ad4d7157dab26e4fb7e/images/day64.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day65
### LLE(Locally linear embedding)
* It is a non-linear dimensionality reduction technique based on mainfold learning unlike PCA which is based on projection.
 ![LLE](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/f1db0434f0d2cff0f3e088c8bf4d1fb1ae50f74c/images/day65%20lle.png)
### K-means
* It is a unsupervised learning algorithm that make clusters of similar data and it has many use cases like image segmentation, anamoly detection, customer segmentation , data analysis and so on.I have already discussed its theory and practical implementation is given below hope you gain some insights.
* ![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dcc9929b5e8b5e8c0a16a4c5764d08f9a7a2dc36/images/day65%20k%20means.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day66
### Supervised learning in Neural network
* Today I get started with new course deep learning specialization and learned about supervised learning using neural network. I learned to implement logistic regression which is classification algorithm using neural network. Traditional algorithm after some amount of data doesnot show increment on performance but with more data you can achieve higher performance by increasing neural network size
* Also learned about backpropagation which is the algorithm use to perform gradient descent in neural network to update the weight and bias in such a way cost function is minimized.
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day67
### Image classifier using sequential api
* Today I learned to use sequential api of keras to create a model that classify mnist fashion dataset clothe images. I trained the model using sequential api with input layer that take image as input and convert it to 1D array and two hidden layer that has relu activation and third output layer with 10 classes and activation softmax for multiple classification and I compile the model and evaluated it and found around 88% accuracy and checked on frist 3 data which was predicted correct.
Here is the code hope you gain some insight watching it.
![day67 done](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1b304b7d047c50b4d2a93cdfbb236587e5149c40/images/day67%20sequential%20api.png)
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day68
* Today I revised concept of vectorization, broadcasting in python and not using of rank 1 array in neural network. Vectorization is done with the help of numpy and it make computation faster by avoiding loop to run in C instead of python. Broadcasting helps to perform arithmetic operation between unsimilar shape vector and also learned to avoid rank 1 array in neural network instead use vectors with rows and column because unexpected broadcasting may happen in rank 1 arrray.These were some basic to clear before diving into deeplearning hope you get some insight from here.
![basic](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/83031bf0abaeeacbf237712531a9d7ec66972031/images/day68%20basic%20of%20deeplearing.png)
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day69
* Today I learned to use functional api to create complex model, I used keras api that formed deep neural network architecture, it contained input layer to take input and the input was normalized by normalization layer, there were two hidden layer with 30 neuron having relu activation and concattenation layer to concatenate both second hidden layer and normalized layer and finally there was hidden layer or output layer to give prediction.
[](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/3483069a043bf3b8346f2924a32b56161a55dd21/images/day69.png)
  ![code](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/3483069a043bf3b8346f2924a32b56161a55dd21/images/day69%20complex%20model%20using%20functional%20api.png)
  * 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day70
* Today I gained some overview on different types of neural network (NN) like ANN(Artifical Neural Network), Convolutional Neural Network (CNN), Recurrent Neural Network(RNN), Auto encoders network , Genrative Adversial Network(GAN).I also learned in detail about perceptron that is mathematical model of neuron and implemented it as binary classifier.
  ![perceptron implementation](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1bb00cf2656eed95fab4d5950bce5acd74b125bf/images/day70%20perceptron.png)
  * 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day71
# Deep learning foundation
### Perceptron trick
* Today I learned about simple perceptron trick in which random data is selected each time and classifer is moved if data is misclassified, but in cannot quantify the classifier and may not converge sometime so we use different lossfunctionn in perceptron
 #### Loss function in Perceptron
 * Different loss function like perceptron loss,hinge loss, relu to minimize loss function in neural network we use gradient descent as done previously in machine learning
### Gradient descent
 * In neural network gradient descent finding is not easy task due to local minima we have to choose correct learning rate to converge so we use a technique called adaptive learnign that chosses learning rate some of them are SGD,adam,RMSprop,etc.
 ### Backpropagation
 * This is done to minimize the loss by updating weights on the basis of gradient descent it utilize chain rule of calculus. In backpropagation we see rate of change of loss with change in weights. It optimizes neural network.
 ### Batching and regularization
 * In neural network trainining is done on batches to enable parallel processing and reduces computation complexity by utilizing vecotrization , it also find more stable gradient and reduces overfitting.
 * Regularization is done on neural network by early stopping and dropout of some neuron.
 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 
|   
[**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day72
### Recurrent neural network (RNNS)
* RNNs are special neural network that takes sequential data like text,audio as input and apply a recurrence relation to process the sequence, RNNs uses same weight matrices everytime ,which allow them to maintain and utilize information from previous time steps while processing the current input.
  ![Rnnsimage](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4ccc60d94814a7804ebf26426c379456393fd307/images/day72%20rnns.png)
### Encoding
* Encoding refers to the process of transforming an input sequence into a fixed-length vector representation or a hidden state that captures the information from the entire sequence.
[encoding](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4ccc60d94814a7804ebf26426c379456393fd307/images/day72%20encoding%20nl.png)
### Embedding 
* Embedding refers to the process of representing categorical or discrete inputs, such as words in natural language processing (NLP), as continuous-valued vectors. Embeddings help capture the semantic relationships and similarities between different input categories.
### Simple RNN example : 
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/4ccc60d94814a7804ebf26426c379456393fd307/images/day72%20rnns_code.png)
 * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day73
### Backpropagation through time (BPTT)
* In RNN backpropagation involves finding the derivatives of the loss function with respect to each time step in the RNN sequence, and then propagating these gradients backwards through time to update the weight called BPTT.
### Gradient Issues
* Computing gradient in rnns can be challenging task because it computing gradient wrt to time state involves many factors of weight repeated gradient computation.
* If weight matrices is large (many values>1) explodint gradients happen and it is solved by:
     * gradient clipping : One common solution to address exploding gradients is gradient clipping, which involves setting a threshold for the gradients. If the gradients exceed the threshold, they are scaled down to maintain their magnitude within an acceptable range.
* If weight matrices is small (many values<1) vanishing gradients happen and it solved by :
  * Activation function : Using activation function like relu make derivative 1 whenever x>0 so it help to prevent vanishing gradient.
  * weight initialization : Using Identity matirces as weight and keeping biases o prevent vanishing problem.
  * Network architecture : Controlling which information to add and which information to remove using  gates in each recurrent network. i.e use of LSTM and GRU.
 * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day74
### LSTM
* Long Short Term Memory(LSTM) it use gated method that selectively add or remove information with each recurrent unit. This architecture solve gradient vanishing problem. It use gate to control flow of information by
   * Forget : Get rid of irrelevant information
   * Store : Store relevant infromation form current input.
   * Update : Selectively update the cell state.
   * Output : Return a filtered version of the cell state.
### Self Attention
* Self attention hold the core concepts behind modern transformer based model . It main idea is `Attending the most important part of input`. It captures long-range dependencies and allow parallelization. 
![self attention image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/7ee81140b9b382df230121902e00b1d966e84368/images/day74%20sellf-attention%20method.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day75
### Convolutional Neural Network(CNN)
CNN are deeplearning algorithm that takes images and videos as input and perfrom further analysis and processing on it trying to mimic human visual system.
* Convolutional layers: In CNN we use filters of features patches to scan images called convolution. Convolution is element wise matrix multiplication of image patch and filter and summing it up which produces feature map. e.g: In letter 8 we use filters of circle,For different feature of image we use different filters.
* Activation functions: After forming feature map we apply activation like Relu to introduce non-linearity in the network.
* Pooling layers: after activation function we use pooling to spatial dimensions which make computational faster, make network more robust for varations in input.
* Flattening : After downsampling of feature map we convert it into 1-D vector and fed it to conncected layer for classification.
* Training : The 1-D data is fed to conncected neural network and model is tarined for further prediction and classification. While training model large sample labelled data is used and model learn through backpropagation.
**To build more robust system we use image augmentation to generate different variation of image for training by scaling,rotating,distorting images**
![cnn](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c9cbf25c21a56b1d4ccdc00cc64912399e99de50/images/day75%20cnn.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day76
* Today I dive depper into each steps of CNN and learned more about parameter of each component and implemented it to create a simple CNNs.
    * I firstly created feature map using Conv2D with 32 filter and filter of 2x2 use relu for non-linearity and downsampled it by using MaxPool2D of size 2x2 and strides i.e movement of 2.
    * In second convolution I used 64 filter of 3x3 with relu for non-linearity and downsampled it using 2x2 pool with stride 2.
    * In last step I flatten the feature map into 1D array and use connected layer to make classification using softmax as activation function.
Hope you get more understanding by looking at code and architecutre of CNN simple implementation.
![cnn_arch](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/3da635804ef77848eb688697c3a34c55a5f21c82/images/day76%20cnn_for_classification.png)
![cnn](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/f4287cacf9beb197e8c90cb73307ee8e182cd10a/images/day75%20cnn_implementation.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day77
### R-CNN
* CNN can only classify one object at a time called image localization to solve this issue sliding window approach was issued which was heavy computationally because it create million of sliding window in a normal image. For solution of this Regional based CNN i.e R-CNN was issued which proposes some region using external algorithm called selective search and Convolution is performed in this region.
* To ensure accurate localization and avoid redundant bounding box proposals, R-CNN applies bounding box regression and a technique called non-maximum suppression (NMS).
![r-cnn](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/90c19139c53a6f08ebeeed45bcd28b74dff63666/images/day77%20rcnn.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day78
### Fast R-CNN and Faster R-CNN
* Fast R-CNN is also object detection algorithm that is faster than R-CNN, because in R-CNN propose region i.e some thousand region were send to CNN but in fast R-CNN input image is send to CNN and after CNN it proposes region from feature map by using selective search algorithm and Region of Interest pooling is done and further prediction is performed.
* But In cases of Faster R-CNN image is send to CNN like fast-RCNN but selective search algorithm is not used instead of it Region based Proposed Network is used to detect object which was faster than selective search algorithm,.
### YOLO algorithm
* You Only Look Once , In this algorithm we take an image and split it into an SxS grid, within each of the grid we take m bounding boxes. For each of the bounding box, the network outputs a class probability and offset values for the bounding box. The bounding boxes having the class probability above a threshold value is selected and used to locate the object within the image.
  ![yolo](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0e952098885f2d7b9b12b450565db67668ea4f23/images/day78%20yolo.webp)
  * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day79
### Bird Species Classification using CNN 
* Today I worked on Bird Species classfication using CNN which take bird image as input convert it to array , normalize it  and passes it CNN which perform classification into one of the 6 classes. My model got an accuracy of around 89% on test data. The model architecutre and snippet is shown below:
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/c5486be98b579cf2953b71d66bb1cb24dd83bb51/images/day79%20cnn_bird_architecture.png)
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/64b231050a74cdf269ca3dbc83eafa76ac147a69/images/day79%20bird_classification.png)
# Day80
### Generative modeling
* Generative modeling take training sample as input with some distribution and predict or generate new sample similar to that distribution.
### Auto encoders
* Automatic encoding or auto encoders simple change higher dimensional data into lower dimension latent space and reconstruct the data using decoder , provides lower dimension data, denoised data.
 ### Variational Auto encoders
 * VAEs are atuo encoders that has probablistic twist on traditional auto encoders, VAEs use mean and standard deviation to learn latent space based on gaussian distribution and decode the latent space to generate new sample data.
![vaes](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/ea57924db99cbb30c1a05f3f5ca0c5eb6afcba86/images/day80%20vaes.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day81
### Prior on the latent disribution
* Prior on the latent distribution means our assumption on how the data will be distibuted for latent variable, common choice will be gaussian distribution with mean= 0 , variance = 1 becaue it encourages even encoding on latent space and penalizes on the clusering of data i.e avoid memorization of data.
### Regularization and normal prior
* In vaes regularzation is done to obatined `continuity : points that are close in latent space are consider similar after decoding` and `completeness: sampling from latent space turn into meaningful content after decoding` .
![regularization and normalprior](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0b285e04491e486991b61ce6708afaacef703858/images/day81%20regularization%20on%20prior.png)
### Reparametrization 
* In vaes reparamerization is done to allow backprop and gradient descent for training of vaes end to end.
  ![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0b285e04491e486991b61ce6708afaacef703858/images/day81%20reparam.png)
### latent peturbation and entanglement
* Keep other variable fixed and increase or decrease single latent variable is latent peturbation and Latent entanglement refers to the interdependencies or correlations among the latent variables in a generative model. In some cases, the latent variables may be entangled, meaning that changing one variable can have an impact on multiple aspects or features of the generated output.
![peturbationn](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0b285e04491e486991b61ce6708afaacef703858/images/day81%20latent-perturbation.png)
![vaes](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/0b285e04491e486991b61ce6708afaacef703858/images/day81%20summaryofvaes.png)
* 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day82
### Genrative Adversial Network (GAN)
![gan](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/20464f2f84d86b7f0a910cf48d30864ef1961c4f/images/day82%20gan_detail.png)
* GAN is made up of two model i.e Generator and Discriminator and they behave adversial role to eachother , where
   * Generator produces synthetic data samples from noise which is similar to real data and Discriminator takes real data and data produced by discriminator as an input and differentiate between fake and real data. Generator and Discriminator works on competitive manner, where generator try to produce good data samples that can fool discriminator and discriminator also improve itself to classify real and fake images. They work on improvement until the data produces by generator becomes as good as real one.
* After training process you can simply use this generator two generate new images that has been never seen before.
 * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day83
### Challenges for robust deeplearning
* Bias skewed data and Uncertainity i.e model doesnot know the answer can be challenging part for developing robust deep learning models.
* **Algorithmic bias** : There can be bias while selection of data i.e some group may be overrepresented while other may be underrepresented, Model bias can be present due to lack of proper benchmark and metrics, Deployment bias due to change in distribution of data overtime, Evaluation bias due to not accounting of subgroups, Intrepretion bias due to human error .
* **Class Imbalance**: Class imablance can be another major problem that create biasness in deeplearning model which doesnot lead to robust model.For example, in a medical diagnosis task, the majority class could be healthy patients, while the minority class could be patients with a rare disease. In fraud detection, the majority class could be non-fraudulent transactions, while the minority class could be fraudulent transactions.
This can be solved by:
![class imbalance solution](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5041d0b9d1a70e4e4904766150c5f875be05792a/images/day83%20classimbalance.png)
 * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day84
### Debiasing VAES
* VAES should be debias for better performance and it is done automatically by increasing the sample probability of sparse region of distribution and undersampling dense region data.
![Debiasing in vaes](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5146635a2cb3230b59bec5f202b1d4854dbbbdf1/images/day84%20vaedebiasing.png)
### Uncertainity
* Uncertainity is the lack of confidence or ambguity in the predictions made by model due to some noise or incomplete data.
 * Aleatoric Uncertainity : It is the uncertainity on data itself due to noise or randomness and it cannot be reduced with more data or model improvement.
 * Epistemic Uncertainity : It is the uncertainity on model  due to incomplete data and can be reduced with more data and model refinement.
![uncertainity type](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/933c5f143d803257ec1c5e3b78c0d5b36b92ee0e/images/day84%20types%20of%20uncertainity.png)
 * 📚Resources
 [Introduction to Deeplearning](https://youtu.be/QDX-1M5Nj7s)
# Day85 
### Auto encoders
* Auto encoders are artificial neural networks capable of learning dense representations of the input data, called latent representations without any supervision. Create an stack autoencoders using keras based on mnist fashion dataset that reconstruct images from latent representations.
Hope you gain some insight from this code about auto_encoders : 
![auto encoders](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/fc9d9dfaaa4dc4f88b069e6c4a594fbf781d0ebb/images/day85%20autoencoders.png)
* Reconstructed image by autoencoder
![result](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/fc9d9dfaaa4dc4f88b069e6c4a594fbf781d0ebb/images/day85%20autoencoder_reconst.png)
# Day86
### Unsupervised retraining using stacked autoencoders
* In realworld we may find data that is unlabeled mostly and few portion is labeled of that data because labeling data is expensive and timeconsuming process , In such case we can train autoencoder on full data in phase1 and train the classification model on labeled data in phase 2 using parameter generated by autoencoder from lower layer.
### Tying weights 
When we have symmetrical autoencoder we can tie the weights of decoder layer to encoder layers this halves the number of weights in the model speeding up training and limiting the overfitting.
![TIE](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/dc2213e56ec286db7f36bb7a4edc5ba12a63ae6d/images/day86%20tiedencoder.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day87
* Today I learned topics like training autoencoders at a time,convolutional autoencoders, denoising autoencoder  from books Hands on Machine learning and implemented them hope you get some insight reading short insights and code snippet.
 * **Training one autoencoder at a time**: We train a first autoencoder with data and the reconstructed data of first autoencoder is sent to second auto encoder , after that the hidden layer of this encoder is stacked and then output layer of this encoder is stacked forminng new stacked autoencoders.
 * **Convolutional autoencoders** : This autoencoders perform well incase of images it reduces the spatial dimensionality of image and increases the depth  i.e feature map  of image. Hope you get some insight watching my Convolutional autoencoder model.
![convoae](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6943328f92cc0da048512213915e98112fc9d37a/images/day87%20convolution%20autoencoder.png)
** **Denoising autoencoders** : Autoencoder can simply be useful to recover noisy image or reconstruct full image by denoising.
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6943328f92cc0da048512213915e98112fc9d37a/images/day87%20denoising_ae-code.png)
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6943328f92cc0da048512213915e98112fc9d37a/images/day87%20denoising_ae.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day88
* Today I learned about sparsity autoencoders from the books and implemented it hope you gain some insights reading this 
* **Sparsity autoencoders** : A sparse autoencoder is a type of neural network that enforces a sparsity constraint on the activations of hidden layer neurons. It encourages most neurons to be inactive, resulting in sparse representations of the input data. By learning meaningful and efficient features, it aids in dimensionality reduction and can be beneficial for various tasks that require feature learning and reconstruction.
![sparisity autoencoders](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/8c826da6ac84d73820c95a50facf0dbfae4b5fcd/images/day88%20sparseautoencoder.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)
# Day89
* Today I learned about variable autoencoders and its implementation on mnist dataset from the book ,  VAEs are atuo encoders that has probablistic twist on traditional auto encoders, VAEs use mean and standard deviation to learn latent space based on gaussian distribution and decode the latent space to generate new sample data. Its implementation is shown belwo hope you gain some insight reading it.
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6ce100ff8c8e19358ea42d0de39f033d78a91fe4/images/day89%20vaep1.png)
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/6ce100ff8c8e19358ea42d0de39f033d78a91fe4/images/day89%20vaep2.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day90
* Today I implemented the Generative Adversarial Network(GAN) from the book using fashion mnist dataset that generated new images of clothes.This model uses a generator to create fake images and a discriminator to distinguish between real and fake images. The GAN is then trained using a custom training loop with alternating training of the discriminator and the generator. The goal is to train the generator to generate realistic images that can fool the discriminator. In this way new realsitic images where generated . Below is the code of gan .
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/f80c5890450e78a12fce59f2998fddc61a11af41/images/day90%20gan.png)
* 📚Resources
  [**Hands-On Machine Learning with Scikit-Learn and TensorFlow**](https://github.com/ageron/handson-ml3)

# Day91 
* Today I revised all the basic concepts like sigmoid function,sigmoid derivative, image to vector conversion, normalizing rows, softmax function, vectorization, L1 loss, L2 loss from course deep learning specilization. Also started to work on project SMS spam classifier using ensemble learning .
![day91_rev](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/d97b907ebf8b81d1b1223488c2d02ce182b22fdf/images/day91%20basic_rev.png)
📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?) 

# Day92 
* Today I revisited my concept on shallow neural network and implemented logistic regression on neural network to create a cat classification model that identify whether the image is cat or not from scratch , I spended time in creating each function that make logistic regression like sigmoid function, initializing weights, learning weights, gradient descent, prediction and model below is the snippet of code hope you gain some insights :
 ![p1](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/9e39ed9eaa5523cf8614fb71fc751e1e5f45f801/images/day92%20logisticp1.png)
![p2](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/9e39ed9eaa5523cf8614fb71fc751e1e5f45f801/images/day93%20logisticp2.png)
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)  
# Day93
* Today I revisited my concept of optimizer in deep learning , basically optmizer main objective is to decrease loss function so that we get higher accuracy , optimization is done with Gradient descent like Batch GD, Stochastic GD and Mini Batch GD . But this optimization has some problem like finding learning rate, rscheduling learning rate , limitation of control of learning rate in multidimension, local minimum , saddle points where slope becomes zero without reaching optimal minimum points.
* EWMA(Exponentially Weighted Moving Average) : It is basically weighted average where past data weight get decreasing over time compare to latest data and it is mostly used with time series data. EWMA can be controlled by a paramter alpha , If we increase alpha It give more value to previous data and graph become more stable, If we give decrease alpha our previous data are less weighted and we get moody graph. Optimal alpah is consider to be 0.9 mostly .
# Day94 
* Today I learned about stochastic gradient descent momentum i.e (SGD momentum) which is used to tackle non convex gradient descent to find global minima , It is extension of SGD  in addition it take a term called velocity which is based on (EWMA)Exponentially Weighted Moving Average of gradient descent it means it move accounting previous gradient descent where latest gradient descent has more weights. It is controlled by a parameter alpha which if 0 act like normal SGD and increasing it increase the velocity , optimal value of this parameter is 0.9 mostly . SGD momentum is lot faster than normal SGD. The problem with this SGD momentum is oscillation of it after reaching global minimum which increases it time to get to optimal solution.
  
![SGD momentum](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/fa3b5ef126469fea947904a59b26a90fb54dad80/images/day94%20sgd_momentum.png)
* Nesterov Accelerated Gradient (NAG) : This optimizers solve the oscillation problem of SGD momentum by damping the oscillation and it become more faster than SGD momentum. It basically update weight in two steps first by momentum and second by gradient or momentum look ahead which reduces oscillation .
![Nag](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/1d137af6368597f5fbf1e6f66a361af8c4f7c83b/images/day94%20nag_momentum.png)
* Adaptive Gradient(ADaGrad) : AdaGrad is an optimizer which use adaptive learning rate and use in case of different scales of features , sparse feature that is due to majority 0 value in feature which create different change in slope in different direction so gradient first move more toward feature that has high gradient update to reduce this learning rate is divided by term which reduces learning rate and make gradient descent convergence faster toward global minimum but due to the term that divides learning rate Gradient descent never converges to global minimum it reaches near it so due this flaw ADAgrad is not used with complex Neural Network It  is only used incase of traditional algorithm but intiuition of it is necessary to learn Adam and RMSprop optimizers.
 
# Day95 
* Today I concluded my optimizers learning by getting deeper intuition about the most use optimizer in deep learning that is Adam optimizer along with one of the best optimizer that can compete with adam i.e RMSprop optimizer.
* RMSprop : This solves the problem of ADaGrad that was not converging to global minimum by changing the learning rate over time. The learning rate has more weight in recent epochs as compare to previous epochs.
* ADAM optimizer : It uses the concept of both momentum and learning rate decay i.e divides learning rate decay by moving average and mulitply learning rate by momentum which makes it use to hill down the gradient descent faster like momentum and get to global minimum without oscillation like learning rate decay concept.
 ![ADAgrad](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/fdfc521b99c750b074f6f42f67c265b59933fc32/images/day94%20adagrad_optimizer.gif)
# Day96
* Today I dive deep into regularization of neural network and got more idea about  L1 and L2 regularization , how regularization reduces overfitting , Dropout regularzation, data augementation , early stopping for reducing overfitting and orthogonalization from course Deeplearning specialization .
* **Regularization** : It is use for reducing overfitting of model most popular way is L2 regularization which penalizes weight for being higher and reduces some of the hidden unit effect. It make model more linear and reduces complex  curve which result in overfitting. L2 doesnot completely make hidden unit zero but penalizes for having higher weight ,
* Whereas in case of **Dropout regularization** It knock out hidden unit randomly making more smaller neural network which reduces the overfitting problem but it has one downside that is it may hamper decreasing cost function so dropout should me made in such a way it doesnot hamper cost function mostly dropout is used with the Computer vision.
* ALso overfitting can be solved by having more data which is achieved by data augmentation.
* Also early stopping at point where devtest cost function stop decreasing reduces overfitting but it hamper gradient descent to reach optimal solution i,e It breaks orthogonalization which means focusing on one task independently without distrubing other i.e gradient descent and regularization.
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day97
* Today I learned about normalizing input features to make gradinet descent faster , Vanishing and exploding of gradients, Checking of gradient descent properly before training model to see if it is working correctly with the help of numerical approximation of gradient i.e Gradient checking and practical implementation of gradient checking from course DeepLearning Initialization , I hope you do the same and here is short summary hope you get some idea: 
 * **Normalizing input features :** Normalizing Input features brings all features in same range and helps gradient desent to converges faster 
 * **Vanishing and exploding gradient** : In deep neural network if weight is too large or too small during back propagation, then weight may explode or  vanish respectively which make deep neural network learning difficult it can be solved with xavier inatilization also known as (glorot) , He initialization  or with other proper initialization of weight .
* **Gradient checking**  : Check the approximated and actual derivative of weight if there difference is 10^-7 then it is good to go if it is more debug. Gradient checking is not used in training , Include regularization term before gradient checking and use gradient checking without dropout and later use dropout. 
Implementation of random and he initialization:
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/778fcbc02f05b589f834fca6edc6ae32dc8b6575/images/day97%20weightinit.png)
* 📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day98
* Today I learned about batch gradient descent,mini-batch gradient descnet and way of choosing mini batch size , bias correction , Hyperparameters and Hyperparamters tuning to find better result from the course Deeplearning initialization and also spended some time implementing regularization from assignment.
* **Mini-batch size** : When training set are too large to be train on one epochs mini batch gradient descent can be used and selecting batch-size depends upon the computational capcity of gpu/cpu. It should not be too large and too small , It is choosen as :
   * If small training set use Batch gd.
   * Typical mini-batch size are in 2^n depending on cpu and gpu size.
   * Many researcher use largest batch size that can fit in gpu
* Where as Bias correction is necessary to handle initial bias in weighted moving average so that initial bias doesnot affect the whole curve.
* **Hyperparameter** : The most important hyperparameter is learning rate ,optimizer,activation function, hidden unit , batch-size,  learning rate decay as well as regularizaition strength i.e L1/L2,dropout rate and initialization method
* Also hyperparameter can be choosen in different method like using coarse to find hyperparamter , picking at random,grid search ,bayesian optimization.
* **Organizing Hyperparamter process** : It is done either by babysitting one model if computational power is limited or training many model at parallel if enough computational power.
*  📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day99
* Today I learned about batch normalization and its whole working process like normalizing activation outputs, fitting batch norm into neural network, batch norm at test time from course deep learning sepcialization and spent some time implementing it .
* **Batch Normalization** : Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. So it advantages are it make network stable,faster,has regularizing effect and due to batch normalization weight initialzation become less important. Batch normalization introduces four parameters: two learnable parameters (alpha and beta) that provide scaling and shifting effects, and two non-learnable parameters (mean and standard deviation). At test time it is done with the help of Exponential Weighted Moving Average (EWMA)  of mean and standard deviation 
* Here is a sample way of applying batch normalization in tensorflow :
![Batch norm](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/90d0717270579f1675e0fe47a776a9358a535511/images/day99%20batch_normalization.png)
* Also you can apply batch normalization after activation function and in some case it may perform well.
*  📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day100
* Today marks my 100 days of code achievement! I've never been this consistent before, and maybe because this field excites me to sit everyday and learn something new , and with each new algorithm I learn, it feels like  gaining new superpowers. Today I learned about structuring and optimizing machine learning projects using orthogonalization,single number evaluation metrics ,optimizing and satisficing metric from course deeplearning specialization.
* **Orthogonalization** : It is an system design property that ensures modification of an component of algorithm doesnot create a side effect to other components.
 For e.g : Early stopping any algorithm to prevent overfitting may create a effect on cost function so this is not consider better interm of orthogonalization.
* **Single Number evaluation metrics:** Evaluating between two different classifier can be bit confusing using precision and recall so single number evaluation metrics for them may be F1 score which can tell which classifier is best.
* **Optimizing and Satisficing metrics** : Optimizing metrics are those metrics that you want to maximize or minimize to get best possible outcome Whereas satisficing is the metrics that are used to ensure certain threshold or minimum requirement is met. In realworld examples combination of optimizing and satisficing metrics is used . for e.g : For any classifier algorithms its optimizing metrics may be F1 score or accuracy and its satisficing metrics may be runtime. 
*  📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day101
* Today I continued my learning on structuring machine learning projects and learned about train/dev/test distribution its size and when to change dev/test sets and its metrics, comparing  with human level perfromance /bayes error and learned to create custom loss and use it  from the course DeepLearning specilization.
* **Managing Train/Dev/Test distribution** : Dev and test set should be of same distribution . Choose a dev set and test set to reflect data you expect to get on future and consider important to do well on.
* **Size of  Test and Dev set** : Set your test set to be big enough to give high conifence in overall performance of your system.
* **Things to remember to move in right direction while working on machine learning projects:**
   * When your metrics is not considering the problem you want to solve tweaks the metrics .
   * Orthogonalization on metrics : Define a metric to evaluate the model, worry separately on how to do well on this metrics.
   * If doing well on dev and test set but doesnot do well on your application change your metrics or dev/test set.
   * E.g : A few ml engineers spending months on tuning their model performance looking to dev set + metrics and when they check on test set with different distrbution it may not perfrom well .
   * Comparing with human level performance : When you surpass human level performance then the accuracy maynot get incrased rapidly and will not go above a line called baysian optimal error. When can follow avoidable bias and avoidable variance tactics by comparing training , test error and Human error proxy as bayesian optimal error. It means that our expectation of model will not be 0% error.
   * Avoidable bias is gap between bayes error/Human level error  and training error and can be controlled by using bigger network,training longer/better optimization. The gap betweeen training error and test error is variance problem and can be controlled by more data, regularization, hyperparameter search.
At last I Learned to create custom loss and create hubber loss that combine both mean squared error (mse) and mean absolute eror (mae) , to use custom loss after loading model we have to provide custom loss you can see from my code below : 
![](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/79df646e444e9f0e2934f33376231a0cf28d5735/images/day101%20custom_model.png)
*  📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day102
* Today I learned about the best practices to do while developing a machine learning applications by reading  and evaluating my understanding from the case study of developing bird detection system in city of pacqueta , error analysis , directing to solve error with the help of misclassified dev set and some healthy habits of machine learning projects development.
* **Error Analysis :** Lets go through a example of cat classifier , If we build a cat classifier and found out few dogs were also classified as cat now what should we do , Should we spend time on dog classifier ?. The answer is ~ check 100 misclassified dev set and count how many are dog . the % of dog in misclassified is the percentage you can get your error down by. IF you got good decrement on error you can spend time on developing dog classifier also .
* Evaluating different mislabelled dev set in parallel : Look out the mislabelled percentage of differnet items and work on making classifier of items that has highest % of misclassification .
* Always remeber to apply changes you made on dev set , to test set so they both have same distribution .
* Also Build your first system quickly and iterate , after that you can work on error analysis and which direction to move for better performance.
Here is a sample of error analysis to get idea [click](https://github.com/Utshav-paudel/Error-analysis-In-Ml-Projects)
![error analysis](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/2815ffe58283bb051be1655e181131a18f00f2de/images/day102%20error_analysis.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day103
* Today I learned about training and testing on different distribution, Bias and variance with mismatched distributions and Addressing data mismatch from the course Deep Learning Specialization .
*  **Training and Testing on different distribution :** This can arise in different machine learning projects , we also choose our test/dev set that represent data distrbution that we will encounter on real world application , dev/test set are the data in which we want to perform better. For e.g : For a cat classifier we can get million of data that are very clear images of cat from internet  but in application user may upload blurry images from camera. So we should make our dev/test set compose of this blurry image from mobile and some amount of this data on training set.
*  In such situation our bias and variance maynot be seen on training error and dev error gap due to difference in data distribution . so we create a new set called `training-dev set which contains training data and dev data, training error and training-dev error will show variance situation`  which help to `see  variance problem` and the `gap between training-dev and dev/test  set will show data mismatch situation `.
*  **Addressing data mismatch :**  Data mismatched problem is addressed by getting more data of test/dev distribution using data synthesis , but during data synthesis be very careful about including tiny subset of data .
# Day104
* Today I learned about transfer learning and multiple task learning and when this technique should be used from the course DeepLearning Specialization.
* **Transfer Learning:** It is  a technique where knowledge learned from a task is reused in order to boost performance on a related task. It is mostly used when you have for the problem you're tranferring from and usually relatively less data for the problem you're transferring to . Also the task should have similar input . for e.g: A cat classifier model trained on 1 million data i.e  pretrainged model is used for dog classfication by finetuning i.e changing output layer or additional layer if needed
* **Multi task Learning:** It is a technique that enables single model to learn multiple taks at once . It is used when training on a set of tasks that could benefit from having share low level features. Usually amount of data is quite similar for each task. for e.g: A object detection that detects car,pedestrian, cycles,etc .
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day105
* Today I learned about end to end deep learning approach and pipepline deeplearning approach and there use cases from course DeepLearning Specilaization.
* **End-to-end deeplearning** : It works well when the data is very large to map input x to the output y and in this approach it may be useful because it lets data speak and doesnot get limited on human perception. Less hand designing of component is needed in this approach.
* **Pipepline approach:** In this approach of deeplearning most important component are only focused for example if you need employee detection system you first take whole picture and keep the face picture and compares it with the orginal dataset to match rather than learning from whole image. It is very useful when large data is not available and It focuses on useful hand-designed component.
* Also Implemented transfer learning to create a classifier that detects simpson cartoon character using VGG16 architecture with some fine tuning.
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/b5543da9e3c1f36678309bb08a473c89319c06d4/images/day105%20tl_p1.png)
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/b5543da9e3c1f36678309bb08a473c89319c06d4/images/day105%20tl_p2.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day106
* Today I completed the prediction portion of Messy or clean room detection using VGG16 architecture by transfer learning, and revised  some computer visions topics like edge detection,padding and strides from course DeepLearning Specialization.
* **Padding** : In normal convolution operation edge are given less emphasis as compared to central area and image shrinks in each convolution to avoid this padding is used which make a external border around the image known as pad which avoid image shrinking and less emphasis on edge. It is of two type : valid means no padding and "same" means pad so output size is same as input.
Here is the some insights of making prediction of model that I saved which was previously trained using transfer learning.
![making prediction](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/bad11b4c36bfbace1dc773572dc780990a294502/images/day106%20tl_p3.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day107
* Today I revised about convolution over volume , notation of convolution layer, Types of layers in CNN, Convolution Neural Network example (LeNet-5) and why convolution instead of neural network from the course DeepLearning Specialization and also spended some time developing cnn from scratch and built zero padding function and forward prop function.
* **Convolution over volume** : When some convolution operation is applied to the input images using some filters then some features are extracted . On a Single image of multi channel i.e RGB image a multichannel vertical edge detection and horizontal edge detection filter is applied which has same channel to images and new Ouput of dimension i.e displayed in images below will be obtained whose channel is equal to the number of filters.
* **Types of Layers in CNN** : Convolution layer, pooling layer, Fully connected layer or Dense layer.
* Input Image is taken an some filter is applied to detect feature for the input image called convolution operation , the layer formed is known as `convolution layer` in such layer `pooling layer` is applied  to reduce dimension this combination of convolution + pooling is taken as 1 layer and many such layer are obtained which generate a final convolution layer that is then flatten and provided to `fully connected layer` and at last require prediction is made.
* **Why Convolution** : Convolution is used instead of simple nn beacause: 
    * Parameter sharing : A feature detector that is used in one part is useful for another part which reduce required number of paramters.
    * Sparsity connections: Output values depends upon only a samll number of input. 
#### Implementation of CNN From scratch 
* Zero padding and convolution single step
* ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/f18a279e7ae0e429cb312e7cf74c50ecb29765c3/images/day107.png)
  
* convolution forward prop
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/f18a279e7ae0e429cb312e7cf74c50ecb29765c3/images/day107%20convolutionp2.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day108 
* Today I Learned about some classical CNN architectures like LeNet-5(60k parameters), AlexNet(~60M parameters), VGG-16(~138M parameters) also learned about ResNets, 1x1 convolution, Inception architecutre, and also created pooling function for CNNs from scratch.
* **ResNet** : The Residual Network (ResNet) allows to train deeper neural network i.e many layer without exploding and vanishing of gradients by skipping connections, while skipping of connection the later activation must match dimension to the previous activation that may be changed by pooling, this matching of activation is done with the help of activation in later activation functions. So with the help of ResNet we can train deeper neural network that bring greater performance which was limited in theory before the invention of ResNet.
* **1*1 convolution** : 1x1 filters are used to decrease the channels , it reduces the channel equivalent to number of 1x1 filters used, which may be helpful since it reduce computational complexity and give more emphasis on important channels.
* **Inception** : The important features of input sometimes get distributed more locally and sometimes get distributed more globally, which may need different filter size to detect them properly, previously same size of filter was used throughout the architecutre but inception stacks different dimensional filter together and to reduce computational complexity it uses 1x1 convolution.
* pooling function to downsampling image
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/641e72ec3cb65dc65e9ee8e1639c8d74017681ad/images/day108%20pooling.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day109
* Today I dive deeper into the intuition of depthwise separable convolution,bottleneck in MobileNetV2, MobileNets , MobileNetsV2, Efficient net from the course Deep Learning Specialization.
*  **MobileNets :** The main idea behind the MobileNets is low computational cost and can be used for mobile device and embedded vision applications. Instead of performing normal convolution MobileNets uses depthwise separable convolutions which is made of depthwise and pointwise convolution which help in reducing number of paramters.
*  **MobileNets V2 :** The mobile net V2 is the improved version of MobileNet which is made of input->expansion(point wise convolution that increases number of channel)-> depthwise convolution that has filters of fxf size and number of filters equal to channel of expanded conv-> projection which is aslo pointwise convolution that decreases channel-> output layer. This middle (i.e expansion->depthwise->projection) is also known as bottle neck reduces computational cost providing good performance in hardware limited scenario like mobile app , embeded system.
*  **Efficient net** : The Effiecient is another architecutre that provides flexibility to change resolution, depth and width of  convNet to get best performance within your computational budget.
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day110
* Today I learned about the open source Implementation of different network archtecture built and using them , transfer learning, data augmentation, state of computer vision from the course deep learning specialization and spended some time implementing image augmentation on a single image.
* **Data augmentation :** The technique use to generate datasets of variation when datasets is limited by applying different transformation like mirroring, random cropping, rotating, shearing, color shifting .  It reduces overfitting problem and increase performance of model.
* State of computer vision : If you have more data than less hand engineering is required and If you have less data then you can use more hand engineering and transfer learning if needed.
Image augmentation implementation
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/a536513f5c9b2672005b448b89700b0a84473b0c/images/day110.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day111
* Today I revisited object detection in deep learning and learned about object localization,landmark detection, Sliding window, YOLO : IoU,Non max suppression,anchor box from the course deep learning specialization.
* **Object Localization:** Detects objects and draw bounding box around them i.e location of the objects.
* **Landmark detection:** It is the process of detecting significant landmark withing the image, these landmark are the point or features of interest within the image such as corners, edges, object keypoints, facial features, or any other salient points that help describe the structure or content of an image.
* **Sliding Window by convolution**: Sliding window according to stride over image may be computationally costly so instead of it we use fully connected layer as convolution and create all possible sliding window at once and select the one with objects but this method maynot give accurate bounding box
* **YOLO algorithm**: This algorithm works well on detecting bounding box than sliding window and it is faster . In this algorithm we take an image and split it into an SxS grid, within each of the grid we take m bounding boxes. For each of the bounding box, the network outputs a class probability and offset values for the bounding box. The bounding boxes having the class probability above a threshold value is selected and used to locate the object within the image.
  * **Intersection Over Union:** IoU measures the overlap between ground truth bounding box and predicted bounding box.
    IoU= Area of Overlap/Area of Union.This IoU makes the preformance of YOLO better by providing more accurate bounding box.
  * **Non-Max suppression:** It selects the bouding box with maximum probability out of different bounding boxes.
  * **Anchor box**: When object of different aspect ratio is in the image then anchor box is used to detect them seperately
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day112
* Today I learned about Image segmentation, transpose matrix multiplication, U-net architecture and spended some time creating a CNN model to detects wether the person is similing and sign language detection using Sequential Api and Functional Api from the course Deep learning specialization.
* **Image Segmentation :** It is the method to label every single pixel instead of drawing bounding boxes in the images and used in automonous driving car, medical field. U-net architecture is one of the popular image segmentation archtitecture . It first use normal convolution and pooling in contraction and in expansion it use transpose convolution with skip connection that provides activation function of higher detailed low level spatial information. Then at last 1x1 convolution filter equal to number of classes to segment is used resulting output with dimension HxW equal to input and 3rd dimension equal to number of class to segment.
* **Functional API vs Sequential API** : Sequential API works very well with linear topology but for non-linear topology you need to use Functional API. I will share some code snippet that reflect way to use Sequential API and Functional API.

**Seqeuntial API use to detect smily face:**
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/0e020a4187780175db3dcee78e89ae1d57a06eee/images/day112%20sequentialapi.png)
**Functional API use to detect sign language:**
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/0e020a4187780175db3dcee78e89ae1d57a06eee/images/day112%20functionalapi.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day113
* Today I dive deep into the concepts behind facial recognition system like one shot learning , siamese network ,  triplet loss from the course deep learning specialization.
* **One-shot learning:** It is a classification algorithm that assesses the similarity and differences between two images. For example for face recognition of a person you get one image from camera and compare it with the image in database . You get some sorts of scores and if the least score is less than threshold ,you are recognized . Thus you don't need to train on thousand of images in different lightings so it is called one shot. This one short learning uses network called siamese network and has loss called triplet loss.
* **Triplet Loss**: Triplet loss is a loss function that recognize the similarity or difference between items. It uses groups of three items called triplets they are : anchor , positive(similar item) and negative(dissimilar item). The loss function encourages the embedding of anchor and positive item to be closer then the embedding of anchor and negative item at least by certain margin.
Implementation of triplet loss :
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/c98f45a4b520e8b2da7b5d7ee4f1eb0dae34937f/images/day113%20tripletloss.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day114
* Today I learned about face recognition technique as binary classification problem, Neural style transfer its cost function , Deep convnet learning from the course Deep learning specialization and spended sometime implementing resnet.
* **Face recognition as binary classification problem :** You have images of verified person in database and you can precompute the embedding of this images and the new images whose face has to be recognized is now embeded and this input image embedding and database embedding is handled by sigmoid function that takes z=Wx+b, where x is the difference of each elements of embedding give 1 if same and 0 if dissimilar face is found.
* **Neural style transfer :** It is the technique in which it have content image and style image , combines them together to generate a new image of content drawn in style refrence.
* At first the generated images is initialize as a  noise and after running gradient descent and minimizing the cost function iteration by iteration the generated image will look more like the rendered image that combines style and content image.
     * **Style cost function:** It is basically the euclidean distance of gram matrix of style image and generated image, where gram matrix is the correlation between activation in k and k' channel where k' means prime channel . Style in any layer is defined as correlation between activations across channels.
     * **Content cost function :** It is basically the euclidean distance between activation of content and activation of generated .
  #### NOTE: combination of content cost and style cost give cost function for Neural style transfer
     * **Deep ConVnet Learning :** The shallower layer detect simpler features like edges, color contrast, corners and deeper layer detect complex features like water , birds legs, people,etc
* Also spendeded some time implementing resnet and created function for identity block that show skip connections.
* ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/1d42fc24febb2d658c2914a8e99ae5d73a6b0745/images/day115%20resnet1.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)

# Day115
* Today I created ResNet-50 from scratch first I built convolutional block function that skips the connection and shortcut path has convolutional layer to equalise dimension of input and output and also used yesterday built identity block that skips connection but it should have same dimension of input and output and it doesnot have convolutional layer on shortcut path. Using this two function I built the entire ResNet-50 model inspired from the course DeepLearning specialization.
* **Convolutional_block : It skips connection , has convolutional layer in shorcut path to make input and output dimension equal**
 ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/1d42fc24febb2d658c2914a8e99ae5d73a6b0745/images/day115%20resnet2.png)
* **ResNet-50 : Built using convolutional_block and identity_block**
* ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/1d42fc24febb2d658c2914a8e99ae5d73a6b0745/images/day115%20resnet3.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day116
* Today I implemented the MobileNet-V2 using transfer learning for image classification task and obtained around 78% accuracy on 5 epochs, since it was the binary classification problem I only used on neuron in dense layer . The overall purpose of using MobileNet-V2 was its tradeoff on accuracy and performance , since it get run in small memory because of depthwise separable convolutions which reduces the number of trainable parameter . Below is the code snippet of transfer learning hope you get some insights reading it , I will put more concise explanation once I complete it fine tuning aiming more accuracy.
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/2af9fba167aef323847dcbb318c925dcb83119e8/images/day%20116_mobilenet.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day117 
* Today I finetuned my MobileNet-V2 , while fine tuning the first layers are for detecting low level simple features like edge,color so changing them is not that important instead you should modify the deeper ending layer because they are made for detecting high level features, I first unfreezed the layer, I started finetuning the layers form 120 to 155 which was last layers and run the 5 more epochs from previous left off and increase the accuracy from 78% to 92 % .
Below is the snippet of code hope you get some insights reading it.
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/f6f5c7cf0d8ee3f161e191002d211e3a59823c1f/images/day%20117mobilenet_finetuning.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day118
* Today I dive deeper into the YOLO model for object detecting , got better understanding about the underhood function like filtering the boxes , Applying Non max suppression to select the best bounding boxes among the confident one , using the IoU threshold to remove overlapped bounding box and overally how the image is converted to encoding by model and the encoding is changed to bounding box that detect objects with scores and classes .
* Below is the code snippet of working of yolo model

Let's breakdown the steps: 
* 1. Your Deep-CNN yolo models take image as input and give you the emedding as output
* 2. We will now take this embedding and convert it to the best bounding box,scores and class name using some of our custom functions.

## Yolo filter boxes : It returns the filter boxes above the threshold from all H_image*W_image*anchor_box ,
## IoU : It is a function used  in non-max suppression to remove boudning box above the IoU threshold.
![](https://github.com/Utshav-paudel/YOLO-Underhood/blob/2f3d8d689d127dcac43dd20cae44f3ca67f10ab3/Image/Yolo%20(1).png)
## Non-Max suppression : It select the most confidence bounding boxes among all the boxes.
![](https://github.com/Utshav-paudel/YOLO-Underhood/blob/2f3d8d689d127dcac43dd20cae44f3ca67f10ab3/Image/Yolo%20(2).png)
## Converting yolo model encoding to the bounding box, scores and class
![](https://github.com/Utshav-paudel/YOLO-Underhood/blob/2f3d8d689d127dcac43dd20cae44f3ca67f10ab3/Image/Yolo%20(3).png)
## Putting all together : Using above function to detect object from the image using pretrained model.
![](https://github.com/Utshav-paudel/YOLO-Underhood/blob/2f3d8d689d127dcac43dd20cae44f3ca67f10ab3/Image/Yolo%20(4).png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day119
* Today I implemented U-net architecture to perform image segmenetation which means labeling each pixel of image, It is basically made of ; 
    * Encoding part : At first image is taken by conv_block() which downsamples the image by performing normal convolution and return next_block and skip_connections, where next_block is used by upcoming convolution that also downsamples and skip_connections is used by corresponding decoding block.
      ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/9a189ddc298de55a7b545b3b00176d19f65cc46c/images/day119%20U-net_encoding.png)
    * Decoding part : It takes previous layer as first parameter and skip connection as second paramter and  performs Transpose convolution to upscale the image and at last convolution the number of filters of convolution is equal to number of classses to be labeled.
    * ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/9a189ddc298de55a7b545b3b00176d19f65cc46c/images/day119%20U-net_decoding.png)
    * U-Net Model : It uses this encoding and decoding to create the segmenetation of image .
      ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/9a189ddc298de55a7b545b3b00176d19f65cc46c/images/day119%20U_net%20final%20model.png)
*   📚Resources
[**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning?)
# Day120
* Today I started LLM from scratch and learned about create own kernel space, setup all the environmen required and learned about word, subword and character tokenizer, where
* Work tokenizer means splitting the raw text into the words base on delimiter this create a huge vocabulary  problem ,  whereas charcter tokenizer splits the raw text into each character this has small vocabulary but single text doesnot provide context which is solved by subword tokenizer which splits the raw text into subword by : not splitting the frequently used small words , but splitting rare large word into small meaningful words.
I also created a character tokenizer that encode and decode each character , I hope you gain some insight from it :
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/d844b2fa280af9c8d734a2fe15cd923816bfd993/images/day120%20tokenizer.py.png)
  * 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day121
* Today I explored the basic operations in pytorch , using of pytorch to process sequential data and learned to properly use the gpu using .to('cuda') when gpu is availabe and checked the difference between gpu and cpu performance in pytroch/
  ![ptyorch_basic](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/9657ed96a6f17757492b7e5448b3d2c225f448c1/images/day121.png)
  * 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day122 
* Today I explored pytorch function and learned about different function like creating sample using torch.multinomial(), concatenation using torch.cat(), creating upper triangular and lower triangular using torch.triu() and torch.tril() respectively, and also about transposing , stacking , masking and mostly the basic building block of  neural network torch.nn and torch.nn.functional ,
* Below is the implementation of all the basic functions of pytorch
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/a444d348c47d7ee476719c0115d0cd5b34a289a4/images/day122%20pytorch_function.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day123
* Today I learned about nn.Embedding and  dived deep into creating Bigram Language model that predicts the next token of word based on previous sequence of token, where model take vocab_size as input .
    * forward pass : It takes index and target = None as  paramter and return logits and loss .
    * generate : It takes index of current context and generate index of current and next context.
 You can get some insights from below code , In coming days I will be finetuning the model created .
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/d46aee4c04b16de90f7eecc88e84c22f904c7527/images/day123%20bigramLanguageModel.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day124
* Today I created the evaluation function and optimized my Bigram Language model, I have used adamw as optimizer and after few k of iterations bigram Language model was able to generate some text that was on similar format to training data.
* BigramLanguge Model : A Bigram Language Model is a type of statistical language model that predicts the probability of a word based on the preceding word in a sequence of words. It is a simple and intuitive approach to language modeling and is often used as a baseline or for educational purposes. After this my bigram Language model was completed hope you get some insights reading this 
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/c29dc9a66514f3ead6d3dd135b6b9426237bc545/images/day124%20bigram_language_eval.png)
  * 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day125 
* Today I revisted some activation like sigmoid, softmax,tanh and dive deeper into the transformer architecutre and learned every detail posssible ,learned about
* **Masked mult head attention** that only provide attention to current and previous token , it doesnot provide attention to next token because in this case model will memorize or overfit and will not learn from output positional encoding so masked is done uisng lower triangular matrix.
* Also spended some time updating my Bigram Language Model to GPTLanguage model and create weight initialization funcition and continued forward pass by providing sequential decoder.
![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/4d98de0931a1c4278e6eb8f63df82ffd633d88c2/images/125%20weight_init%20and%20forward%20pass.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day126
* Today I created decoder block of my GPT language model Which simply is like -> Multihead attention -> add and normalize -> feedforward -> add and normalize ->  after the creation of Block class then I created  the feedforward class that is used in decoder block which is simply like Linear-> ReLU -> Linear . Below is the code snippet of Decoder Block and its feedforward portion.
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/326887652f0bc8a4a0dc16a608e7911f38987bb0/images/day126%20decoder_block%2Cfeedforward%20layer.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day127 
* Today I created Multihead attention class and Block class , where multi head attention is used to provide the realtion of each word with others , it provides dependecies of token or word and most importantly I used  nn.Modulelist() for head which make them run parallely making multihead attention faster .
* Below is the code snippet for this Multihead attention.
* Head
  ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/046726571cd7b8887ffbd84282687f2dc7c4c80c/images/day127%20Head.png)
* Multi-Head attention
* ![](https://github.com/Utshav-paudel/300DaysOFMachineLearning-DeepLearning/blob/046726571cd7b8887ffbd84282687f2dc7c4c80c/images/day127%20Multiheadattention.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
# Day128
* Today I continued to work on my Large Language model to create a system that will feed the large corpus of text data to LLM for training which was openwebtext data with around 20k separate files, So I write my script that was able to read all file in format .xyz and append them into list.
* From list of .xyz file I splited them into parts each part containing few file
* Loop was run to each part and the vocab of the part was updated i.e set of characters was updated.
* each character was store in separate line.
 ![](https://github.com/Utshav-paudel/NLP-Transformer-LLM-fever/blob/31cff1117b5a7a3b127f743af06e407a6e251ad1/LLM_from_scratch/day9/data_extractionn_from_text_corpus.png)
![](https://github.com/Utshav-paudel/NLP-Transformer-LLM-fever/blob/31cff1117b5a7a3b127f743af06e407a6e251ad1/LLM_from_scratch/day9/part2_to_store_new_vocabulary.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
  
# Day129 
* Today I updated my data feeding scripts where I created system to process training and validation data seperately which seem more effective than storing the whole corpus on single file and dividing while training it will be more inefficient , learned about authenicating admin rights while running script that need admin rights like extracting of whole 20k files.
![](https://github.com/Utshav-paudel/NLP-Transformer-LLM-fever/blob/0c82f50e2821c33c6851c30b338b28a55f89e385/LLM_from_scratch/day10/extracting_trainandval_seperately.png)
* 📚Resources
    [**LLM from Scratch**](https://youtu.be/UU1WVnMk4E8?si=2r58NpTIn-Z0Y_Z2)
