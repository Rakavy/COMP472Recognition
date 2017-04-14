# Import the modules
import cv2
import numpy as np

from collections import Counter
import importData
from sklearn import ensemble, svm, metrics
from skimage import feature
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


data = importData.getData()

def getHog(img):
    H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True)
    return H


####### Get Traning and Testing Set #################

trainingSet = importData.getTrainingSet()
testingSet = importData.getTestingSet()

print("Count of digits in training", Counter(trainingSet[1]))
print("Count of digits in testing", Counter(testingSet[1]))

hog_data = []
n_samples = len(trainingSet[0])
for f in trainingSet[0]:  
    fd = getHog(f)
    hog_data.append(fd)
hog_features = np.array(hog_data).reshape((n_samples, -1))

expected = testingSet[1]
hog_data_test = []

n_samples_test = len(testingSet[0])
for f in testingSet[0]:  
    fd = getHog(f)
    hog_data_test.append(fd)
hog_features_test = np.array(hog_data_test).reshape((n_samples_test, -1))

print("---SVM CLASSIFICATION---")

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(hog_features, trainingSet[1])

score=classifier.score(hog_features_test, expected)
print('Score\t'+str(score))

predicted = classifier.predict(hog_features_test)

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(testingSet[0], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()  

print("---RANDOM FOREST CLASSIFICATION---")
# #Using the Random Tree Classifier
classifier = ensemble.RandomForestClassifier()

# #Fit model with sample data
classifier.fit(hog_features, trainingSet[1])

# #Attempt to predict validation data
predicted = classifier.predict(hog_features_test)

score=classifier.score(hog_features_test, expected)
print('Score\t'+str(score))

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(testingSet[0], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()  

print("---DECISION TREE CLASSIFICATION---")

# Decision Tree Classifier
classifier = DecisionTreeClassifier()

# #Fit model with sample data
classifier.fit(hog_features, trainingSet[1])

# #Attempt to predict validation data
predicted = classifier.predict(hog_features_test)

score=classifier.score(hog_features_test, expected)
print('Score\t'+str(score))

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(testingSet[0], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()  

print("---NAIVE BAYES CLASSIFICATION---")
# Naive Bayes Classifier
classifier = GaussianNB()

##Fit model with sample data
classifier.fit(hog_features, trainingSet[1])

# #Attempt to predict validation data
predicted = classifier.predict(hog_features_test)

score=classifier.score(hog_features_test, expected)
print('Score\t'+str(score))

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(testingSet[0], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()