# python 2 

import numpy
import nupic
from nupic.bindings.algorithms import SpatialPooler

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

digits = load_digits()

binary_digits = load_digits()

# Now, let's threshold the data
threshold = 12
#binary_digits = numpy.copy(digits)
binary_digits.data[digits.data < threshold] = 0.0
binary_digits.data[digits.data > 0.0] = 1.0

#x_train, x_test, y_train, y_test = train_test_split(binary_digits.data, binary_digits.target, test_size=0.25, random_state=0)
x_train = binary_digits.data[0:1347]
y_train = binary_digits.target[0:1347]
x_test = binary_digits.data[1348:-1]
y_test = binary_digits.target[1348:-1]

sp = SpatialPooler(inputDimensions=(8,8),
					columnDimensions=(16,16),
					potentialRadius=4,
					potentialPct=0.5,
					globalInhibition=False,
					localAreaDensity=0.03, #0.02,
					numActiveColumnsPerInhArea=-1.0,
					stimulusThreshold=2,
					synPermInactiveDec=0.008,
					synPermActiveInc=0.05,
					synPermConnected=0.30,
					minPctOverlapDutyCycle=0.05,
					dutyCyclePeriod=500,
					boostStrength=4.0,
					seed=-1,
					spVerbosity=1,
					wrapAround=False)

def visualizeSDR(inputImage, SDR, pauseInterval=2.0):
	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.imshow(inputImage, cmap=plt.cm.gray)
	ax2.imshow(SDR, cmap=plt.cm.gray)
	plt.pause(pauseInterval)


def visualizeReceptiveFields(column1, column2, column3, column4, sp, numInputs=64, pauseInterval=10.0):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

	# get receptive fields
	R1 = numpy.zeros(numInputs)
	R2 = numpy.zeros(numInputs)
	R3 = numpy.zeros(numInputs)
	R4 = numpy.zeros(numInputs)
	sp.getConnectedSynapses(column1,R1)
	sp.getConnectedSynapses(column2,R2)
	sp.getConnectedSynapses(column2,R2)
	sp.getConnectedSynapses(column2,R2)
	ax1.set_title(str(column1))
	ax2.set_title(str(column2))
	ax3.set_title(str(column3))
	ax4.set_title(str(column4))
	ax1.imshow(R1.reshape((8,8)), cmap=plt.cm.gray)
	ax2.imshow(R3.reshape((8,8)), cmap=plt.cm.gray)
	ax3.imshow(R3.reshape((8,8)), cmap=plt.cm.gray)
	ax4.imshow(R4.reshape((8,8)), cmap=plt.cm.gray)
	plt.pause(pauseInterval)

# train
N_train = x_train.shape[0]
i = 0
for image in x_train[0:N_train]:
	image = image.astype(numpy.uint32)
	#SDR = numpy.zeros((16,16))
	SDR = numpy.zeros(256)
	SDR = SDR.astype(numpy.uint32)
	sp.compute(image, True, SDR)
	if i % 500 == 0:
		print "training"
		visualizeSDR(image.reshape((8,8)), SDR.reshape((16,16)))
	i += 1

# now turn off sp learning and generate an sp SDR dataset
#sp._learn = False

#SDR_train = numpy.zeros((1347,256))
SDR_train = numpy.zeros((x_train.shape[0], 256))
i = 0
for image in x_train:
	image = image.astype(numpy.uint32)
	SDR = numpy.zeros(256)
	SDR = SDR.astype(numpy.uint32)
	sp.compute(image, False, SDR)
	SDR_train[i] = SDR.astype(float) #.reshape(-1)
	i += 1


#SDR_test = numpy.zeros((450,256))
SDR_test = numpy.zeros((x_test.shape[0], 256))
i = 0
for image in x_test:
	image = image.astype(numpy.uint32)
	SDR = numpy.zeros(256)
	SDR = SDR.astype(numpy.uint32)
	sp.compute(image, False, SDR)
	SDR_test[i] = SDR.astype(float)
	i += 1

# now train and test classifier
LRmodel = LogisticRegression()
LRmodel.fit(SDR_train, y_train)

predictions = LRmodel.predict(SDR_test)

score = LRmodel.score(SDR_test, y_test)
print "accuracy = " + str(score)

# visualize duty cycles
dutycycles = numpy.zeros((16,16))
sp.getActiveDutyCycles(dutycycles)
#plt.close()

plt.figure()
plt.imshow(numpy.reshape(dutycycles, (16,16)), cmap=plt.cm.gray)
plt.pause(5.0)
plt.close()


# show some example receptive fields
connectedCounts = numpy.zeros(256)
sp.getConnectedCounts(connectedCounts)

#print(connectedCounts)

# visualize some receptive fields
visualizeReceptiveFields(13,84,20,59, sp)