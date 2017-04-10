import cv2
import os
import numpy as np
import pylab as pl 
import imutils
from skimage import feature

images = []
labels = []

new_width = 64
new_height = 64

def getData():
    for i in range(0,10):
        for filename in os.listdir('../data/'+str(i)):
            img = cv2.imread(os.path.join('../data/'+str(i),filename))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgH = cv2.resize(gray, (new_width, new_height)) 
                 
                images.append(imgH)
                labels.append(i)
    return [images,labels]

def getTrainingSet():
    trainingFeature = []
    trainingLabel = []

    for i in range(0,500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])
    
    for i in range(1000,1500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])
    
    for i in range(2000,2500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(3000,3500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(4000,4500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(5000,5500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(6000,6500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(7000,7500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(8000,8500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    for i in range(9000,9500):
        trainingFeature.append(images[i])
        trainingLabel.append(labels[i])

    return (trainingFeature, trainingLabel)

def getTestingSet():
    testingFeature = []
    testingLabel = []

    for i in range(500,1000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])
    
    for i in range(1500,2000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])
    
    for i in range(2500,3000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(3500,4000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(4500,5000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(5500,6000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(6500,7000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(7500,8000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(8500,9000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])

    for i in range(9500,10000):
        testingFeature.append(images[i])
        testingLabel.append(labels[i])
 

    return (testingFeature, testingLabel)
