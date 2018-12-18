# python 2 

import numpy
import nupic
from nupic.bindings.algorithms import SpatialPooler, TemporalMemory
import matplotlib.pyplot as plt
import random

cellsPerColumn = 5
tp = TemporalMemory(
               columnDimensions=(200,),
               cellsPerColumn=cellsPerColumn,
               activationThreshold=2,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=1,
               maxNewSynapseCount=5,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               maxSegmentsPerCell=25,
               maxSynapsesPerSegment=25,
               seed=42)

sp = SpatialPooler(inputDimensions=(50,),
					columnDimensions=(200,),
					potentialRadius=15,
					potentialPct=0.5,
					globalInhibition=True,
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

# Define my sequences
encoded_A = numpy.zeros(50)
mask_A = numpy.array(list(range(10)))
encoded_A[mask_A] = 1
encoded_A = encoded_A.astype(numpy.uint32)

encoded_B = numpy.zeros(50)
mask_B = mask_A + 10
encoded_B[mask_B] = 1
encoded_B = encoded_B.astype(numpy.uint32)

encoded_C = numpy.zeros(50)
mask_C = mask_B + 10
encoded_C[mask_C] = 1
encoded_C = encoded_C.astype(numpy.uint32)

encoded_D = numpy.zeros(50)
mask_D = mask_C + 10
encoded_D[mask_D] = 1
encoded_D = encoded_D.astype(numpy.uint32)

print "A =", str(encoded_A)
print "B =", str(encoded_B)
print "C =", str(encoded_C)
print "D =", str(encoded_D)

codes = [encoded_A, encoded_B, encoded_C, encoded_D]

# create sequences
uniqueSequences = [[3,1,2,3], [0,1,2,0]]
seqLength = 4

def generateBalancedSequenceSample(num_seqs=2, sample_size=100):
	'''
	num_seqs = number of unique sequences to choose from
	sample_size = number of sequence id's to choose (i.e. the number of sample sequences)
	'''
	seqs = []
	for s in range(sample_size):
		seq_id = random.randint(0,num_seqs-1)
		seqs.append(seq_id)
	return numpy.array(seqs)



# define the SDR's for the 4 letters
SDR_A = numpy.zeros(200)
SDR_A = SDR_A.astype(numpy.uint32)
sp.compute(encoded_A, False, SDR_A)
SDR_A_activeColumns = numpy.where(SDR_A > 0)[0]
SDR_A_activeColumns.sort()

SDR_B = numpy.zeros(200)
SDR_B = SDR_B.astype(numpy.uint32)
sp.compute(encoded_B, False, SDR_B)
SDR_B_activeColumns = numpy.where(SDR_B > 0)[0]
SDR_B_activeColumns.sort()

SDR_C = numpy.zeros(200)
SDR_C = SDR_C.astype(numpy.uint32)
sp.compute(encoded_C, False, SDR_C)
SDR_C_activeColumns = numpy.where(SDR_C > 0)[0]
SDR_C_activeColumns.sort()

SDR_D = numpy.zeros(200)
SDR_D = SDR_D.astype(numpy.uint32)
sp.compute(encoded_D, False, SDR_D)
SDR_D_activeColumns = numpy.where(SDR_D > 0)[0]
SDR_D_activeColumns.sort()

print "SDR_A_activeColumns", SDR_A_activeColumns
print "SDR_B_activeColumns", SDR_B_activeColumns
print "SDR_C_activeColumns", SDR_C_activeColumns
print "SDR_D_activeColumns", SDR_D_activeColumns

SDR_codes = [SDR_A, SDR_B, SDR_C, SDR_D]
SDR_activity_codes = [SDR_A_activeColumns, SDR_B_activeColumns, SDR_C_activeColumns, SDR_D_activeColumns]


# create sample populations of sequences for training the temporal pooler
train_size = 1000
seqs_train = generateBalancedSequenceSample(num_seqs=len(uniqueSequences), sample_size=train_size)

# This will hold n-1 SDR's for each sequence in the sample population of sequences (where n = seqLength is the length of each sequence in the population of sequences); the final spatial pooler SDR will not be included because we want the temporal pooler to predict this
SP_SDR_seqs_train = []


for i, s_id in enumerate(seqs_train):
	s = uniqueSequences[s_id]
	#s = s[0:-1]
	#SP_SDR_train = numpy.zeros((seqLength,200))
	SP_SDR_train = numpy.zeros((seqLength,6))
	SP_SDR_train = SP_SDR_train.astype(numpy.uint32)
	for j, symbol in enumerate(s):
		#print "symbol=", symbol
		#print "j=", j
		SP_SDR_train[j] = SDR_activity_codes[symbol]
	SP_SDR_seqs_train.append(SP_SDR_train)

	# now train temporal memory
	for j, SDR in enumerate(SP_SDR_train):
		tp.compute(SDR, learn=True)

	# how do we reset the tp ?
	tp.reset()


# Now, check predictions of tp
def printTemoralPredictions(SP_activeCol_seq, symbol_seq):
	for i, s in enumerate(SP_activeCol_seq):
		tp.compute(s, learn=False)
		print "TP Winner cells", tp.getWinnerCells()
		print "TP Predictive cells", tp.getPredictiveCells()
	tp.reset()
	print "\n"

def cellIndexToCoordinates(index):
	column = tp.columnForCell(index)
	cell = index - (column+1)*cellsPerColumn
	return (cell, column)

def visualizeTemporalPooler(SP_activeCol_seq, symbol_seq):
	for i, s in enumerate(SP_activeCol_seq):
		tp.compute(s, learn=False)
		activeCells = tp.getWinnerCells()
		predictiveCells = tp.getPredictiveCells()
		print "TP Winner cells", activeCells
		print "TP Predictive cells", predictiveCells
		#activeColumns = []
		#predictiveColumns = []
		#for cell in activeCells:
		#	col = tp.column
		tp_activity = numpy.zeros((cellsPerColumn, 200))
		for cell in activeCells:
			position = cellIndexToCoordinates(cell)
			tp_activity[position] = 2
		for cell in predictiveCells:
			position = cellIndexToCoordinates(cell)
			if tp_activity[position] == 2:
				tp_activity[position] = 3
			else:
				tp_activity[position] = 1

		#tp_activity[activeCells] = 
		tp_activity = tp_activity.reshape(-1)
		plt.figure()
		plt.imshow(tp_activity.reshape((cellsPerColumn*4, 50)))
		plt.savefig("temporal_activity_step_" +str(i) + ".png")
		plt.pause(3.0)
	tp.reset()
	print "\n"

def getSPactiveColSeq(seq):
	SP_activeCol_seq = []
	for symbol in seq:
		SP_activeCol_seq.append(SDR_activity_codes[symbol])
	return SP_activeCol_seq

def getAllSPactiveColSeqs(uniqueSequences):
	allseqs = []
	for s in uniqueSequences:
		seq = getSPactiveColSeq(s)
		allseqs.append(seq)
	return allseqs

uniqueSPactiveColSeqs = getAllSPactiveColSeqs(uniqueSequences)

#printTemoralPredictions(uniqueSPactiveColSeqs[0], uniqueSequences[0])

visualizeTemporalPooler(uniqueSPactiveColSeqs[0], uniqueSequences[0])


#printTemoralPredictions(uniqueSPactiveColSeqs[1], uniqueSequences[1])







