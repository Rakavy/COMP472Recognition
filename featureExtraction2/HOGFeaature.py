# Import the modules
import cv2
import numpy as np

from collections import Counter
import importData
from sklearn import ensemble, svm, metrics
from skimage import feature


data = importData.getData()

def getHog(img):
    H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True)
    return H


###### Get Traning and Testing Set #################

trainingSet = importData.getTrainingSet()
testingSet = importData.getTestingSet()

print "Count of digits in training", Counter(trainingSet[1])
print "Count of digits in testing", Counter(testingSet[1])



hog_data = []
n_samples = len(trainingSet[0])
for f in trainingSet[0]:  
    fd = getHog(f)
    hog_data.append(fd)
hog_features = np.array(hog_data).reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(hog_features, trainingSet[1])

# Now predict the value of the digit on the second half:
expected = testingSet[1]
hog_data_test = []

n_samples_test = len(testingSet[0])
for f in testingSet[0]:  
    fd = getHog(f)
    hog_data_test.append(fd)
hog_features_test = np.array(hog_data_test).reshape((n_samples_test, -1))

predicted = classifier.predict(hog_features_test)

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))