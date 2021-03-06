# Classifying digits code based on http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

import matplotlib.pyplot as plt
from sklearn import ensemble, svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import glob, os
import numpy as np
from PIL import *
import pylab as pl 

def loadDigits(digit):
	new_width = 64
	new_height = 64
	
	list = []
	for file in glob.glob("data/" + str(digit) + "/*"):
		im = Image.open(file)
		im = im.resize((new_width, new_height), Image.ANTIALIAS)
		list.append(np.array(im))
	return list

def main():
	persianDigits = []
	persianLabels = []
	images_and_labels = []
	for num in range(0,10):
		digits = loadDigits(num)
		print("Feature Extraction for num: " + str(num))
		persianDigits.extend(digits)
		persianLabels.extend([num] * len(digits))

	pl.gray() 
	pl.matshow(persianDigits[13]) 
	pl.show() 	

	images_and_labels = list(zip(persianDigits, persianLabels))
	for index, (image, label) in enumerate(images_and_labels[:15]):
	    plt.subplot(3, 5, index + 1)
	    plt.axis('off')
	    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	    plt.title('%i' % label)

	# To apply a classifier on this data, we need to flatten the image, to
	# turn the data in a (samples, feature) matrix:
	n_samples = len(persianDigits)
	setDivision = int(0.8 * n_samples)
	from sklearn.utils import shuffle
	persianDigits, persianLabels = shuffle(persianDigits, persianLabels, random_state=0)
	data = np.array(persianDigits).reshape((n_samples, -1))

	print("---SVM CLASSIFICATION---")

	# Create a classifier: a support vector classifier
	classifier = svm.SVC(gamma=0.001)

	# We learn the digits on the first half of the digits
	classifier.fit(data[:setDivision], persianLabels[:setDivision])

	# Now predict the value of the digit on the second half:
	expected = persianLabels[setDivision:]
	predicted = classifier.predict(data[setDivision:])

	score=classifier.score(data[setDivision:], expected)
	print('Score\t'+str(score))

	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	images_and_predictions = list(zip(persianDigits[setDivision:], predicted))
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
	classifier.fit(data[:setDivision], persianLabels[:setDivision])

	expected = persianLabels[setDivision:]
	predicted = classifier.predict(data[setDivision:])

	score=classifier.score(data[setDivision:], expected)
	print('Score\t'+str(score))

	print("Classification Method: RFC \n")
	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	images_and_predictions = list(zip(persianDigits[setDivision:], predicted))
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
	classifier.fit(data[:setDivision], persianLabels[:setDivision])

	expected = persianLabels[setDivision:]
	predicted = classifier.predict(data[setDivision:])

	score=classifier.score(data[setDivision:], expected)
	print('Score\t'+str(score))

	print("Classification Method: DTC \n")
	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	images_and_predictions = list(zip(persianDigits[setDivision:], predicted))
	for index, (image, prediction) in enumerate(images_and_predictions[:4]):
	    plt.subplot(2, 4, index + 5)
	    plt.axis('off')
	    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	    plt.title('Prediction: %i' % prediction)

	plt.show()


	print("---NAIVE BAYES CLASSIFICATION---")
	# Naive Bayes Classifier
	classifier = GaussianNB()

	# #Fit model with sample data
	classifier.fit(data[:setDivision], persianLabels[:setDivision])

	expected = persianLabels[setDivision:]
	predicted = classifier.predict(data[setDivision:])

	score=classifier.score(data[setDivision:], expected)
	print('Score\t'+str(score))

	print("Classification Method: NBC \n")
	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	images_and_predictions = list(zip(persianDigits[setDivision:], predicted))
	for index, (image, prediction) in enumerate(images_and_predictions[:4]):
	    plt.subplot(2, 4, index + 5)
	    plt.axis('off')
	    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	    plt.title('Prediction: %i' % prediction)

	plt.show()

main()