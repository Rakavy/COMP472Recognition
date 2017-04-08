# Import the modules
import cv2
import numpy as np

from collections import Counter
import importData

# Load the dataset and Extract the features and labels
data = importData.getData()
features = np.array(data[0]) 
labels = np.array(data[1])

SZ=20
bin_n = 16 # Number of bins

svm_params = dict(kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0,ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1,ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


# Extract the hog features
list_hog_fd = []
for feature in features:  
    fd = hog(feature)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print "Count of digits in dataset", Counter(labels)

###### Get Traning and Testing Set #################

trainingSet = importData.getTrainingSet()
testingSet = importData.getTestingSet()

print "Count of digits in training", Counter(trainingSet[1])
print "Count of digits in testing", Counter(testingSet[1])


######     Now training      ########################

#deskewed = [map(deskew,row) for row in train_cells]

list_hog_fd = [] 
for feature in trainingSet[0]: 
    fd = hog(feature) 
    list_hog_fd.append(fd) 
hog_features = np.array(list_hog_fd, 'float64') 

trainData = np.float32(hog_features).reshape(-1,64)
responses = np.array(trainingSet[1])

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setDegree(0.0)
svm.setGamma(5.383)
# svm.setCoef0(0.0)
svm.setC(2.67)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')


######     Now testing      ########################

#deskewed = [map(deskew,row) for row in test_cells]

list_hog_fd_test = [] 
for feature in testingSet[0]: 
    fd = hog(feature) 
    list_hog_fd_test.append(fd) 
hog_features_test = np.array(list_hog_fd_test, 'float64') 

testData = np.float32(hog_features_test).reshape(-1,bin_n*4)
result = svm.predict(testData)

#######   Check Accuracy   ########################
mask = result==testingSet[1]
correct = np.count_nonzero(mask)
print (correct*100.0/len(result))