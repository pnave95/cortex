import numpy
from BASIC_spatial_pooler import SpatialPooler

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
binary_digits = load_digits()


threshold = 12
binary_digits.data[digits.data < threshold] = 0.0
binary_digits.data[digits.data > 0.0] = 1.0

def visualizeSDR(inputImage, SDR, pauseInterval=2.0):
	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.imshow(inputImage, cmap=plt.cm.gray)
	ax2.imshow(SDR, cmap=plt.cm.gray)
	plt.pause(pauseInterval)



if __name__ == '__main__':

	# Create a spatial pooler object
	inputDimensions = (8,8)
	columnDimensions = (16,16)
	potentialRadius = 5
	potentialPct = 0.5
	sparsity = 0.10
	stimulusThreshold = 2
	synPermInactiveDec = 0.008
	synPermActiveInc = 0.05
	synConnectedPermThreshold = 0.15
	learn=True
	seed=12345
	debug=False
	sp = SpatialPooler(inputDimensions, columnDimensions, potentialRadius, potentialPct, sparsity, stimulusThreshold, synPermInactiveDec, synPermActiveInc, synConnectedPermThreshold, learn, seed, debug)

	# get image data
	imageData = binary_digits.data
	print(type(imageData))
	print(imageData.shape)

	# Visualize learning
	N = 1797
	for i in range(N):
		image = imageData[i]
		#print(type(image))
		#print(image.shape)
		sp.compute(image.reshape((8,8)))
		SDR = sp._columnActivations
		if i % 100 == 0:
			print(i)
			visualizeSDR(image.reshape((8,8)), SDR.reshape((16,16)), 2.0)



