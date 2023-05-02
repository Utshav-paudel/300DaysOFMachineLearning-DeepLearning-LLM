![machine learning image](https://github.com/Utshav-paudel/MachineLearning-DeepLearning/blob/5b5aa0e37fc4f1f2904987b7cd6bd018398c9f16/images/ml%20and%20dl.avif)
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
