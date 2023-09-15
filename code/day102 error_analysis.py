#@ Implementation of error analysis 
from sklearn.datasets import load_digits           # downscaled version of mnist digit
digits = load_digits()
#@ Viz of datasets
from matplotlib import pyplot as plt
plt.figure(figsize = (8,16))
_, axes = plt.subplots(2,5)
images_and_labels = list(zip(digits.images, digits.target))               # assigning images and labels           
for ax, (image, label) in zip(axes.flatten(), images_and_labels[:10]):    # viz of 10 images
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Digit {label}")
# separating data into training and dev set 
from sklearn.model_selection import train_test_split

#split data into train and test subsets
n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)
X_train, x_dev, y_train, y_dev = train_test_split(
data, digits.target, test_size=0.5, shuffle=False)
# traing a softmax classifier 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter = 10000)

# Training the model
classifier.fit(X_train,y_train)

# predicting on dev set
predict = classifier.predict(x_dev)

(predict == y_dev).mean() # checking accuracy 

# viz of some misclassification and correct classificaiton

# dev set digits that are misclassified 
X_error = x_dev[predict != y_dev]     
y_error = y_dev[predict != y_dev]     
p_error = predict[predict != y_dev]

# dev set digits that are classified correctlty
X_corr = x_dev[predict == y_dev]
y_corr = y_dev[predict == y_dev]
p_corr = predict[predict == y_dev]

#show the histogram
plt.xticks(range(10))
plt.hist(y_error)