import numpy
from BASIC_spatial_pooler import SpatialPooler

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

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
	synConnectedPermThreshold = 0.10
	dutyCyclePeriod=100
	boostStrength=1.0
	learn=True
	seed=12345
	debug=False
	sp = SpatialPooler(inputDimensions, columnDimensions, potentialRadius, potentialPct, sparsity, stimulusThreshold, synPermInactiveDec, synPermActiveInc, synConnectedPermThreshold, dutyCyclePeriod, boostStrength, learn, seed, debug)

	# get image data
	imageData = binary_digits.data
	print(type(imageData))
	print(imageData.shape)

	# test / train split
	x_train, x_test, y_train, y_test = train_test_split(binary_digits.data, binary_digits.target, test_size=0.25, random_state=0)

	# Visualize SDR training
	N_train = 1347
	i = 0
	for image in x_train:
		sp.compute(image.reshape((8,8)))
		SDR = sp._columnActivations
		if i % 200 == 0:
			print(i)
			visualizeSDR(image.reshape((8,8)), SDR.reshape((16,16)), 2.0)
		i += 1

	# classification learning
	SDR_train = numpy.zeros((N_train, 256))
	sp._learn = True
	k = 0
	for image in x_train:
		sp.compute(image.reshape((8,8)))
		SDR = sp._columnActivations
		SDR_train[k] = SDR
		if k % 200 == 0:
			print(k)
			visualizeSDR(image.reshape((8,8)), SDR.reshape((16,16)), 2.0)
		k += 1

	# visualize classification learning


	# Visualize learning
	'''
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
	'''

	print("Final permanences = ", sp._permanences)



