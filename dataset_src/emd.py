'''
Created on 11/04/2015

@author: Andrew Chalmers

This code computes the Earth Mover's Distance, as explained here:
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm

This is done using numpy, scipy (minimize)

There is a simple example of two distributions computed by getExampleSignatures()
This example is chosen in order to compare the result with a C implementation 
found here:
http://robotics.stanford.edu/~rubner/emd/default.htm
'''

import numpy as np
import scipy.optimize
import scipy.spatial as spa
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt

# Constraints
def positivity(f):
	'''
	Constraint 1: 
	Ensures flow moves from source to target
	'''
	return f 

def fromSrc(f, wp, i, shape):
	"""
	Constraint 2: 
	Limits supply for source according to weight
	"""
	fr = np.reshape(f, shape)
	f_sumColi = np.sum(fr[i,:])
	return wp[i] - f_sumColi

def toTgt(f, wq, j, shape):
	"""
	Constraint 3: 
	Limits demand for target according to weight
	"""
	fr = np.reshape(f, shape)
	f_sumRowj = np.sum(fr[:,j])
	return wq[j] - f_sumRowj

def maximiseTotalFlow(f, wp, wq): 
	"""
	Constraint 4: 
	Forces maximum supply to move from source to target
	"""
	return f.sum() - np.minimum(wp.sum(), wq.sum())

# Objective function
def flow(f, D):
	"""
	The objective function
	The flow represents the amount of goods to be moved 
	from source to target
	"""
	f = np.reshape(f, D.shape)
	return (f * D).sum()

# Distance
def groundDistance(x1, x2, norm = 2):
	"""
	L-norm distance
	Default norm = 2
	"""
	return np.linalg.norm(x1-x2, norm)

# Distance matrix
def getDistMatrix(s1, s2, norm = 2):
	"""
	Computes the distance matrix between the source
	and target distributions.
	The ground distance is using the L-norm (default L2 norm)
	"""
	# Slow method
	# rows = s1 feature length
	# cols = s2 feature length
	numFeats1 = s1.shape[0]
	numFeats2 = s2.shape[0]
	distMatrix = np.zeros((numFeats1, numFeats2))

	for i in range(0, numFeats1):
		for j in range(0, numFeats2):
			distMatrix[i,j] = groundDistance(s1[i], s2[j], norm)

	# Fast method (requires scipy.spatial)
	#import scipy.spatial
	#distMatrix = scipy.spatial.distance.cdist(s1, s2)

	return distMatrix
	
# Flow matrix
def getFlowMatrix(P, Q, D):
	"""
	Computes the flow matrix between P and Q
	"""
	numFeats1 = P[0].shape[0]
	# print(numFeats1)
	numFeats2 = Q[0].shape[0]
	shape = (numFeats1, numFeats2)
	
	# Constraints  
	cons1 = [{'type':'ineq', 'fun' : positivity},
			 {'type':'eq', 'fun' : maximiseTotalFlow, 'args': (P[1], Q[1],)}]
	
	cons2 = [{'type':'ineq', 'fun' : fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
	cons3 = [{'type':'ineq', 'fun' : toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]
	
	cons = cons1 + cons2 + cons3

	# Solve for F (solve transportation problem) 
	F_guess = np.zeros(D.shape)
	F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
	F = np.reshape(F.x, (numFeats1,numFeats2))
	
	return F

# Normalised EMD
def EMD(F, D):  
	"""
	EMD formula, normalised by the flow
	"""
	return (F * D).sum() / F.sum()
  
# Runs EMD program  
def getEMD(P,Q, norm = 2):
	"""
	EMD computes the Earth Mover's Distance between
	the distributions P and Q
	
	P and Q are of shape (2,N)
	
	Where the first row are the set of N features
	The second row are the corresponding set of N weights
	
	The norm defines the L-norm for the ground distance
	Default is the Euclidean norm (norm = 2)
	"""  
	
	D = getDistMatrix(P[0], Q[0], norm)
	F = getFlowMatrix(P, Q, D)
	
	return EMD(F, D)
	
# Examples
def getSignatures1(input1,input2):
	"""
	returns signature1[features][weights], signature2[features][weights]
	"""
	# features1 = np.array([[100, 40, 22], 
	# 					  [ 211, 20, 2 ], 
	# 					  [ 32, 190, 150 ], 
	# 					  [ 2, 100, 100 ] ])
	features1 = np.array(input1)
	weights1 = np.array(np.ones(len(input1)))


	# features2 = np.array([ [ 0, 0, 0 ], 
	# 					   [ 50, 100, 80 ], 
	# 					   [ 255, 255, 255 ] ])
	features2 = np.array(input2)
	weights2 = np.array(np.ones(len(input2)))
	
	signature1 = (features1, weights1)
	signature2 = (features2, weights2)
	
	return signature1, signature2

def getExampleSignatures2():
	"""
	returns signature1[features][weights], signature2[features][weights]
	"""
	features1 = np.array([[100, 40, 22], 
							[ 211, 20, 2 ], 
							[ 32, 190, 150 ], 
							[ 2, 100, 100 ] ])
	weights1 = np.array([ 1.0,1.0,1.0,1.0 ])


	features2 = np.array([ [ 0, 0, 0 ], 
							[ 50, 100, 80 ], 
							[ 255, 255, 255 ] ])
	weights2 = np.array([ 1.0,1.0,1.0 ])
	
	signature1 = (features1, weights1)
	signature2 = (features2, weights2)
	
	return signature1, signature2

def getExample_GaussianHistograms(N = 15, showPlot = True):
	"""
	returns signature1[features][weights], signature2[features][weights]
	"""
	x = np.linspace(-1,1,N)
	y1 = np.exp(-np.power(x - 0.0, 2.0) / (2 * np.power(0.2, 2.0)))
	y2 = np.exp(-np.power(x - 0.5, 2.0) / (2 * np.power(0.2, 2.0)))
	y1 /= np.sum(y1)
	y2 /= np.sum(y2)

	if showPlot:
		plt.bar(x, y1, width=0.1, alpha=0.5)
		plt.bar(x, y2, width=0.1, alpha=0.5)

	#features1 = y1.reshape((N,1))
	#weights1 = (1.0/N) * np.ones((N))

	#features2 = y2.reshape((N,1))
	#weights2 = (1.0/N) * np.ones((N))
	
	#signature1 = (features1, weights1)
	#signature2 = (features2, weights2)

	signature1 = (x.reshape((N,1)), y1.reshape((N,1)))
	signature2 = (x.reshape((N,1)), y2.reshape((N,1)))
	
	return signature1, signature2

def doRubnerComparison(input1,input2):

	# Setup
	P, Q = getSignatures1(input1,input2)

	# Get EMD
	emd = getEMD(P, Q)
	
	# Output result
	# print('We got: '+str(emd))
	return emd
	# print('Rubner C example got 160.54277')

def doGaussianHistogramExample():
	showPlot = True

	# Setup
	P, Q = getExample_GaussianHistograms(showPlot=showPlot)

	# Get EMD
	emd = getEMD(P, Q)
	
	# Output result
	print('EMD: '+str(emd))

	if showPlot:
		plt.show()


if __name__ == '__main__':
	print('EMD')
	coor1=[]
	coor2=[]
	c4=[]
	c4_=[]
	#doGaussianHistogramExample()
	biopython_parser = PDBParser()
	in_str="./dataset_src/cpeb3/7QR4_1_B.pdb"
	structure = biopython_parser.get_structure("random_id",in_str)
	model = structure[0]
	for i,chain in enumerate(model):
		print(i)
		# print(j)
		# c_=model[0]
		# print(model[0])
		for res_idx, residue in enumerate(chain):

			print(res_idx)
			print(residue)
			if res_idx==1:
				rs1=residue
				for atom in rs1:
					coor1.append(list(atom.get_vector()))
					if atom.name == 'C4\'':
						# c4=list(atom.get_vector())
						c4.append(list(atom.get_vector()))
			if res_idx==2:
				rs2=residue
				for atom in rs2:
					coor2.append(list(atom.get_vector()))
					if atom.name == 'C4\'':
						# c4_=list(atom.get_vector())
						c4_.append(list(atom.get_vector()))
	print(rs1)
	print(rs2)

	print(coor1)
	# print(coor2)
	coor1=np.array(coor1).T
	coor2=np.array(coor2).T
	print(coor1)

	# print(c4)
	# print(c4_)

	distances = spa.distance.cdist(c4, c4_)
	print(distances)
	doRubnerComparison(coor1,coor2)
	print('Success')	