from sklearn import cluster, datasets
import glob, os
from PIL import Image
from skimage import feature
import numpy as np

def loadDigits():
    new_width = 64
    new_height = 64	
    images = []
    labels_samp = []
    for i in range(0,10):
        for file in glob.glob("data/" + str(i) + "/*"):
            im = Image.open(file)
            if im is not None:
                im = im.resize((new_width, new_height), Image.ANTIALIAS)
                images.append(im)
                labels_samp.append(i)
    return [images,labels_samp]

def getHog(img):
    H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True)
    return H


digits = loadDigits()

hog_data = []
n_samples = len(digits[0])
for f in digits[0]:  
    fd = getHog(f)
    hog_data.append(fd)
data = hog_data
#iris = datasets.load_iris()
#X_iris = iris.data
#y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(data) 
cluster.KMeans(algorithm='auto', copy_x=True, init='k-means++')
print data
print digits[1]
print(k_means.labels_)
print(digits[1])
