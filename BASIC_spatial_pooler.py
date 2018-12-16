import numpy
import random
import itertools

# Not used
class CorticalColumns:
	"""
	This class represents cortical column matrices.  This class will be used in three different ways: (1) It will hold potential neighborhood adjacency matrices, (2) it will hold synapse permanences, and (3) it will hold binary connection information
	"""
	def __init__(self, dataMatrix, columnDimensions):
		assert(len(dataMatrix.shape) == 2)  # make sure we are actually being passed a matrix
		self._columnDimensions = columnDimensions
		self._shape = tuple(columnDimensions)
		self._numInputs = dataMatrix.shape[1]
		self._numColumns = dataMatrix.shape[0]
		self._columns = dataMatrix

	def __getitem__(self,index):  # rows = cortical columns
		return self._columns[index]


	def getColumn(self, *indices):
		if len(indices)==1:
			return self._columns[indices]
		else:
			assert(len(indices)==len(self._shape))
			flatindex = numpy.ravel_multi_index(numpy.array(indices), self._shape)
			return self._columns[flatindex]

	def setColumn(self, column, *indices):
		if len(indices)==1:
			self._columns[indices] = column
		else:
			assert(len(indices)==len(self._shape))
			flatindex = numpy.ravel_multi_index(numpy.array(indices), self._shape)
			#return self._columns[flatindex]
			self._columns[flatindex] = column

# Not used
class PotentialNeighborhoods(CorticalColumns):
	"""
	The main data of this class 
	"""
	pass


def indexToCoords(flatindex, shape):
	return numpy.unravel_index(flatindex, shape)

def coordsToIndex(coords, shape):
	return numpy.ravel_multi_index(numpy.array(coords), shape)

def getNeighborhood(centerIndex, radius, dimensions):
	'''
	This function takes the index of a point within a multidimensional array and computes the indices of all the neighboring points
	Args:
		centerIndex (int):  The (flat) index of the point at the center of the neighborhood
		radius (int):  The radius of the neighborhood (l_infinity metric)
		dimensions (indexable sequence):  An array of ints representing the dimensions of the multidimensional array we are working in
	Returns:
		(numpy array):  The indices of all points in the neighborhood (including centerIndex)
	'''
	# get coordinates of the center point
	centerCoords = indexToCoords(centerIndex, dimensions)

	# get intervals for each dimension of hypercube neighborhood 
	intervals = []
	for i, dimension in enumerate(dimensions):
		left = max(0, centerCoords[i] - radius)
		right = min(dimension - 1, centerCoords[i] + radius)
		intervals.append(list(range(left, right+1)))

	# itertools.product() computes the Cartesian product of iterables.  In this case,
	# we compute the Cartesian product of several intervals to get the coordinates of
	# all points in our neighborhood
	neighborCoords = numpy.array(list(itertools.product(*intervals)))

	# convert our array of neighbor positions into an array of neighbor indices
	#print(neighborCoords)
	neighborIndices = coordsToIndex(neighborCoords.T, dimensions)

	return neighborIndices 


def array2arrayCoordinateProjection(columnCoordinates, columnDimensions, inputDimensions):
	'''
	Given coordinates describing a column and the dimensions of the column space, we rescale the column coordinates to lie between 0 and 1; then we rescale again to fit input dimensions and finall round to get the nearest point in the input space; we return the coordinates of this point
	'''
	relativeCoordinates = []
	for i, L in enumerate(columnDimensions):
		x = columnCoordinates[i] / (L-1)
		relativeCoordinates.append(x)

	inputCoordinates = []
	for i, L in enumerate(inputDimensions):
		x = int(round(relativeCoordinates[i] * (L-1)))
		inputCoordinates.append(x)

	return numpy.array(inputCoordinates)
	

def array2arrayIndexProjection(columnIndex, columnDimensions, inputDimensions):
	columnCoordinates = indexToCoords(columnIndex, columnDimensions)
	inputCoordinates = array2arrayCoordinateProjection(columnCoordinates, columnDimensions, inputDimensions)
	return coordsToIndex(inputCoordinates, inputDimensions)


class RandomVariable:
	'''
	Primary method:  sample
	'''
	pass

class ContinuousRandomVariable(RandomVariable):
	'''
	Primary methods:  pdf, cdf, sample
	If no cdf is known, it will be estimated via integration
	'''
	pass

class SpatialPooler:

	def __init__(self,
				 inputDimensions=(8,8),
				 columnDimensions=(16,16),
				 potentialRadius=4,
				 potentialPct=0.5,
				 sparsity=0.02,
				 stimulusThreshold=2,
				 synPermInactiveDec=0.008,
				 synPermActiveInc=0.05,
				 synConnectedPermThreshold=0.1,
				 dutyCyclePeriod=1000,
				 boostStrength=0.0,
				 learn=True,
				 seed=12345,
				 debug=False):

		random.seed(seed)  # initialize with known seed for reproducibility

		columnDimensions = numpy.array(columnDimensions, ndmin=1)
		numColumns = columnDimensions.prod()

		inputDimensions = numpy.array(inputDimensions, ndmin=1)
		numInputs = inputDimensions.prod()

		self._numInputs = numInputs
		self._numColumns = numColumns

		self._inputDimensions = inputDimensions
		self._columnDimensions = columnDimensions

		self._potentialRadius = potentialRadius
		self._potentialPct = potentialPct
		self._sparsity = sparsity
		self._stimulusThreshold = stimulusThreshold
		self._synPermInactiveDec = synPermInactiveDec
		self._synPermActiveInc = synPermActiveInc
		self._synConnectedPermThreshold = synConnectedPermThreshold

		self._synBelowStimulusPermInc = 0.05
		self._dutyCyclePeriod = dutyCyclePeriod

		#self._flatDataShape = (numInputs, numColumns)
		# rows are cortical columns, columns are inputs
		self._flatDataShape = (numColumns, numInputs)

		self._potentialConnections = self.getPotentialConnections((numColumns, numInputs), columnDimensions, inputDimensions, potentialRadius)

		Permanences = self.initPermanences()
		self._permanences = Permanences
		self._connectedSynapses = self.initConnections(Permanences, synConnectedPermThreshold)

		self._activePotentialSynapses = numpy.zeros((numColumns, numInputs))
		self._activeConnectedSynapses = numpy.zeros((numColumns, numInputs))

		self._rawColumnActivations = numpy.zeros(numColumns)
		self._boostedColumnActivations = numpy.zeros(numColumns)
		self._uninhibitedColumnActivations = numpy.zeros(numColumns)
		self._columnActivations = numpy.zeros(numColumns)

		self._boostFactors = numpy.ones(numColumns)
		activeDutyCycles = numpy.ones(numColumns)
		#self._activeDutyCycles = activeDutyCycles.fill(sparsity)
		self._activeDutyCycles = self.initActiveDutyCycles(sparsity, numColumns)
		#print("activeDutyCycles", activeDutyCycles)

		self._boostStrength = boostStrength
		self._boostFactors = numpy.ones(numColumns)

		self._learn = learn 

		self._debug = debug

		# Create a numpy iterator to iterate through permanences during learning
		#perm_iter = numpy.nditer(self._)



	#def getPotentialConnections(self):
	def getPotentialConnections(self, flatDataShape, columnDimensions, inputDimensions, potentialRadius):
		# return the 2D array with 1's for points in potential nbd and 0's elsewhere
		potentialConnections = numpy.zeros(flatDataShape)
		for columnIndex, column in enumerate(potentialConnections):
			# project column into input space
			projectedInputIndex = array2arrayIndexProjection(columnIndex, columnDimensions, inputDimensions)
			potentialMask = getNeighborhood(projectedInputIndex, potentialRadius, inputDimensions)
			column[potentialMask] = 1.0
			potentialConnections[columnIndex] = column

		return potentialConnections

	def initActiveDutyCycles(self, sparsity, numColumns):
		activeDutyCycles = numpy.ones(numColumns)
		activeDutyCycles.fill(sparsity)
		return activeDutyCycles

	def initPermanences(self):
		#permanences = numpy.zeros(self._flatDataShape)

		# describe the shape of the distribution to sample from
		a = 0.0
		m = self._synConnectedPermThreshold
		b = min(1.0, 2*m)
		# currently, we are assuming potentialPct = 0.5
		permanences = numpy.random.triangular(a, m, b, self._flatDataShape)
		return numpy.multiply( self._potentialConnections, permanences)


		#mask = PotentialNeighborhoods.nonzero()  # get locations of all potential synapses
		#nonpotentials = numpy.where(PotentialNeighborhoods == 0)
		#self._permanences[nonpotentials] = 0

	def initConnections(self, permanences, synConnectedPermThreshold):
		mask = numpy.where(permanences >= synConnectedPermThreshold)
		ConnectedSynapses = numpy.zeros(permanences.shape)
		ConnectedSynapses[mask] = 1
		return ConnectedSynapses

	def matrixRowOverlap(self, vector, matrix):
		vector = vector.reshape(-1)
		assert(len(vector) == len(matrix[0]))

		for rowIndex, row in enumerate(matrix):
			row = numpy.multiply(vector, row)
			matrix[rowIndex] = row
		return matrix


	def updateActiveSynapses(self, inputVector):
		inputVector = inputVector.reshape(-1)
		self._activePotentialSynapses = self.matrixRowOverlap(inputVector, self._potentialConnections)
		self._activeConnectedSynapses = self.matrixRowOverlap(inputVector, self._connectedSynapses)

		if self._debug == True:
			print("activeConnectedSynapses = \n", self._activeConnectedSynapses)

	def updateRawColumnActivations(self):
		#sum each row
		self._rawColumnActivations = numpy.sum(self._activeConnectedSynapses, 1)

	def updateBoostedColumnActivations(self):
		self.updateRawColumnActivations()
		self._boostedColumnActivations = numpy.multiply(self._rawColumnActivations, self._boostFactors)

	def getMostActiveColumnsGlobal(self):
		'''
		Compute the top sparsity*numColumns columns by activity
		'''

		# get raw column activations (number of synapses active for each column).  Once boosting is incorporated into this model, we will not use raw activations but instead boosted activations (perhaps reversing the order of normalizing and boosting also)
		self.updateBoostedColumnActivations()
		#boostedActivations = numpy.multiply(self._rawColumnActivations, self._boostFactors)


		# Get the indices that would sort rawColumnActivations
		#indices = numpy.argsort(self._rawColumnActivations)
		indices = numpy.argsort(self._boostedColumnActivations)
		indices = numpy.flip(indices,0)


		#compute max number of active columns
		maxActiveColumns = max(int(round(self._sparsity * self._numColumns)),1)

		if self._debug == True:
			print("rawColumnActivations\n",self._rawColumnActivations)
			print("boostedColumnActivations:\n", self._boostedColumnActivations)
			print("maxActiveColumns = ", maxActiveColumns)

		# check if we need to do any inhibition
		self.updateUninhibitedColumnActivations()
		#if maxActiveColumns >= numpy.sum(self._uninhibitedColumnActivations):
		if maxActiveColumns >= len(numpy.where(self._boostedColumnActivations >= self._stimulusThreshold)[0]):
			return numpy.copy(self._uninhibitedColumnActivations)

		# compute the smallest number of synapses which still puts a column into the top active columns
		#minMaxActivity = self._rawColumnActivations[indices[maxActiveColumns-1]]
		minMaxActivity = self._boostedColumnActivations[indices[maxActiveColumns-1]]

		# get all uncontested activated column indices
		#maxIndices = numpy.argwhere(self._rawColumnActivations > minMaxActivity)
		maxIndices = numpy.argwhere(self._boostedColumnActivations > minMaxActivity)
		maxIndices = maxIndices[:,0]

		# check if there is a tie
		#nextHighestActivation = self._rawColumnActivations[indices[maxActiveColumns]]
		nextHighestActivation = self._boostedColumnActivations[indices[maxActiveColumns]]
		if nextHighestActivation < minMaxActivity:
			activations = numpy.zeros(self._numColumns)
			#activations[self._rawColumnActivations >= minMaxActivity] = 1
			activations[self._boostedColumnActivations >= minMaxActivity] = 1
			return activations 


		# get array of all contested activated column indices
		#contested = numpy.argwhere(self._rawColumnActivations == minMaxActivity)
		contested = numpy.argwhere(self._boostedColumnActivations == minMaxActivity)
		contested = contested[:,0]

		# randomly select however many more columns we need
		activations = numpy.zeros(self._numColumns)
		activations[maxIndices] = 1
		needed_columns = max(0,maxActiveColumns - len(maxIndices))
		contested = contested.tolist()

		if self._debug == True:
			print("needed_columns = ", needed_columns)
			print("guaranteed activations = \n", activations)
			print("contested = \n", contested)

		for i in range(needed_columns):
			add_index = random.choice(contested)
			del contested[contested == add_index]
			activations[add_index] = 1
		return activations


		

	def updateUninhibitedColumnActivations(self):
		# no boosting, so this is the same as just thresholding and normalizing the raw activations
		#self.updateRawColumnActivations()
		A = numpy.zeros(self._numColumns)
		A[self._boostedColumnActivations >= self._stimulusThreshold] = 1
		self._uninhibitedColumnActivations = A

	def updateColumnActivations(self):
		# no inhibition currently, so the same as uninhibited activations
		#self.updateUninhibitedColumnActivations()
		#self._columnActivations = numpy.copy(self._uninhibitedColumnActivations)
		self._columnActivations = self.getMostActiveColumnsGlobal()

	def updatePermanences(self):
		# update permanences
		updates = -self._synPermInactiveDec * numpy.copy(self._potentialConnections)
		for rowIndex, row in enumerate(updates):
			row = row + (self._synPermInactiveDec + self._synPermActiveInc) * self._columnActivations[rowIndex]
			updates[rowIndex] = row

		self._permanences = self._permanences + updates

	def updateConnections(self):
		mask = numpy.where(self._permanences >= self._synConnectedPermThreshold)
		self._connectedSynapses = numpy.zeros(self._flatDataShape)
		self._connectedSynapses[mask] = 1

	def learn(self):
		self.updatePermanences()
		self.updateConnections()

		if self._debug == True:
			print("activeDutyCycles: ",self._activeDutyCycles)
			print("columnActivations", self._columnActivations)
			print("dutyCyclePeriod = ", self._dutyCyclePeriod)

		self._activeDutyCycles = self.updateDutyCyclesHelper(self._activeDutyCycles, self._columnActivations, self._dutyCyclePeriod)
		#self._activeDutyCycles = activeDutyCycles
		self.updateBoostFactorsGlobal()

		if self._debug == True:
			print("boostFactors:\n",self._boostFactors)


	def compute(self, inputVector):
		self.updateActiveSynapses(inputVector)

		self.updateColumnActivations()
		
		if self._learn == True:
			self.learn()

	def updateDutyCyclesHelper(self, dutyCycles, nextInput, windowSize):
		assert(windowSize >= 1)
		return (dutyCycles * (windowSize - 1.0) + nextInput) / windowSize

	def updateBoostFactorsGlobal(self):
		targetDensity = self._sparsity
		self._boostFactors = numpy.exp( (targetDensity - self._activeDutyCycles) * self._boostStrength)

	#def raisePermanenceToThreshold(self, )




if __name__ == '__main__':

	# test spatial pooler
	
	sp = SpatialPooler(inputDimensions=(7),
				columnDimensions=(14),
				potentialRadius=1,
				potentialPct=0.5,
				sparsity=0.02,
				stimulusThreshold=2,
				synPermInactiveDec=0.008,
				synPermActiveInc=0.05,
				synConnectedPermThreshold=0.1,
				dutyCyclePeriod=1000,
				boostStrength=0.1,
				learn=True,
				seed=12345,
				debug=True)
	
	#print(sp._potentialConnections)
	numpy.set_printoptions(precision=2)
	#print(sp._permanences)
	#print(sp._connectedSynapses)

	# Test overlap computation
	inputVector = numpy.array([0,1,1,1,0,0,1])
	sp.compute(inputVector)
	columnActivations = sp._columnActivations
	print("columnActivations = \n", columnActivations)


	# Test neighborhood finding
	'''
	A = numpy.array(list(range(25)))
	A = A.reshape((5,5)) #+ 4
	print("A =\n",A)
	index = 13
	position = indexToCoords(index,A.shape)
	radius = 1

	neighborhood = getNeighborhood(index,radius,A.shape)
	print("neighborhood = \n", neighborhood)

	B = A.reshape(25)
	print("B=\n", B)
	B[neighborhood] = 1
	B = B.reshape((5,5))
	print("neighborValues=\n",B)
	'''