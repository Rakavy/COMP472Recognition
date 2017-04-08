import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random
from sklearn import ensemble
import pandas as pd
import glob, os
import numpy as np
from PIL import Image
import pylab as pl 

def loadDigits(digit):
	list = []
	for file in glob.glob("data/" + str(digit) + "/*"):
		im = Image.open(file)
		list.append(np.array(im))
	return list




def main():
	digits = load_digits() 	
	persianDigits = {'0' : [], '1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : []}
	# for num in range(0,9):
	# 	persianDigits[str(num)] = loadDigits(num)
	# print(len(persianDigits['0']))
	# print(digits.images[0])
	# print("=========================")
	# print(persianDigits['0'][0])
	# print(persianDigits['0'][0])
	
	pl.gray() 
	# pl.matshow(persianDigits['0'][0]) 
	pl.matshow(digits.images[0]) 
	pl.show() 	

	digits.images[0] 	

	print(digits.target)
	images_and_labels = list(zip(digits.images, digits.target))
	for index, (image, label) in enumerate(images_and_labels[:15]):
	    plt.subplot(3, 5, index + 1)
	    plt.axis('off')
	    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	    plt.title('%i' % label)

	#Define variables
	n_samples = len(digits.images)
	x = digits.images.reshape((n_samples, -1))
	y = digits.target

	#Create random indices 
	print(len(x))
	sample_index=random.sample(range(len(x)),len(x)/5) #20-80
	valid_index=[i for i in range(len(x)) if i not in sample_index]

	#Sample and validation images
	sample_images=[x[i] for i in sample_index]
	valid_images=[x[i] for i in valid_index]

	#Sample and validation targets
	sample_target=[y[i] for i in sample_index]
	valid_target=[y[i] for i in valid_index]

	#Using the Random Tree Classifier
	classifier = ensemble.RandomForestClassifier()

	#Fit model with sample data
	classifier.fit(sample_images, sample_target)

	#Attempt to predict validation data
	score=classifier.score(valid_images, valid_target)
	print('Random Tree Classifier:\n') 
	print('Score\t'+str(score))


	i=150

	pl.gray() 
	pl.matshow(digits.images[i]) 
	pl.show() 
	classifier.predict(x[i])

main()