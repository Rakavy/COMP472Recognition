import numpy as np
from sklearn.cluster import KMeans,DBSCAN ,SpectralClustering,Birch
import glob, os
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from itertools import cycle
from skimage import feature
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import metrics

def loadDigits(digit):
	new_width = 64
	new_height = 64
	
	list = []
	for file in glob.glob("../data/" + str(digit) + "/*"):
		im = Image.open(file)
		im = im.resize((new_width, new_height), Image.ANTIALIAS)
		list.append(np.array(im))
	return list

def getHog(img):
    H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True)
    return H

def main():
    persianDigits = []
    persianLabels = []
    images_and_labels = []
    for num in range(0,10):
        digits = loadDigits(num)
        persianDigits.extend(digits)
        persianLabels.extend([num] * len(digits))

    n_samples = len(persianDigits)

    hog_data = []
    for f in persianDigits:  
        fd = getHog(f)
        hog_data.append(fd)
    data = np.array(hog_data).reshape((n_samples, -1))
    SepectralClustering(data,persianLabels)

def SepectralClustering(data,actualLabels):
    pca = PCA(n_components=2).fit(data)
    pca_2d = pca.transform(data)
    spectral = SpectralClustering(n_clusters=10,eigen_solver='arpack',affinity="nearest_neighbors")
    t0 = time()
    spectral.fit(pca_2d)
    print('% 9s' % 'init'
      '    time   homo   compl  v-meas     ARI AMI  silhouette')
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f'   
            %('Spectral', (time() - t0),
            metrics.homogeneity_score(actualLabels, spectral.labels_),
            metrics.completeness_score(actualLabels, spectral.labels_),
            metrics.v_measure_score(actualLabels, spectral.labels_),
            metrics.adjusted_rand_score(actualLabels, spectral.labels_),
            metrics.adjusted_mutual_info_score(actualLabels,  spectral.labels_),
            metrics.silhouette_score(data, spectral.labels_,metric='euclidean',sample_size=10000)))

    print spectral.labels_
    print len(np.unique(spectral.labels_))
    colors = np.random.rand(15)
    scatter = plt.scatter(pca_2d[:,0],pca_2d[:,1],c=spectral.labels_, marker='*')

    plt.colorbar(scatter)

    plt.title('Spectral Clustering')
    plt.show()

main()