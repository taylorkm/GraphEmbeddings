import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors as nnc
from scipy.sparse import coo_matrix
import csv # for reading and writing to CSV
import os.path
import matplotlib.pyplot as plt # for visualizing results
from mpl_toolkits.mplot3d import Axes3D # for 3D plots




# Read data from a file that must be formatted as:
#
# Header 1, Header 2, ..., Header N
#      x11,      x12, ...,      x1N
#      x21,      x22, ...,      x2N
#      x31,      x32, ...,      x3N
def readFromFile( filename = 'data.csv' ):
	if ( not os.path.isfile(filename) ):
		print('File does not exists.')
		return 

	with open(filename, 'r') as f:
		reader = csv.reader(f)		
		data   = [map(lambda x: float(x), row) for row in reader]
		# each element of data is a list, representing a row of the file
			
	return np.array(data)


# Write data to CSV file, each row represents a new datapoint.
def writeToFile( iterable_data_object, filename = 'data.csv' ):
	if ( os.path.isfile(filename) ):
		print('File already exists.')
		return 

	with open(filename, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(iterable_data_object)


def buildGraphLaplacian( data, sigma, nnn ):
	# INPUT:
	# data is a N-by-d matrix of N points in d dimensions
	#
	# nnn is the number of nearest neighbors to use when 
	# constructing the graph
	#
	# If two nodes i and j are nearest-neighbors and they
	# are a distance dist(i,j) apart, then the edgeweight 
	# (before symmetrizing) is w(i,j) = exp(- d(i,j)**2 / sigma**2 )
	# 
	# OUTPUT:
	# A N-by-N matrix representing the normalized graph
	# Laplacian. It's eigenvalues 0=\lambda_0 < \lambda_1 < ...
	# are associated with the eigenvectors that oscillate
	# the least over the graph.

	distToWeight = lambda d, sigma: np.exp( - (d / sigma) ** 2 )

	# Get nearest neighbor indices
	N = len(data) # number of data points	
	nbhds = nnc(n_neighbors=nnn, algorithm='ball_tree').fit( data )
	distances, col = nbhds.kneighbors(data)
	row = np.tile(range(len(data)),(nnn,1) ).T

	# Build weight matrix
	w = coo_matrix( 
		(
			distToWeight(distances.flatten(), sigma),
			(row.flatten(), col.flatten() )
		),
		shape=(N, N)
	).toarray()

	# symmetrize
	w = (w+w.T)/2
	d = np.sum( w, axis = 1)

	d = np.sqrt( np.sum( w, axis = 1) )
	l = np.dot( np.diag( 1./d ) , w )
	l = np.dot( l, np.diag( 1./d ) )

	return np.eye(N) - l

	
# Build a swiss roll dataset.
def buildSwissRoll( number_of_samples = 1500 ):
	t = np.linspace(0, np.pi*4, number_of_samples )[:-1]
	x = (1-t/ np.pi / 5)*np.cos(t)
	y = (1-t/ np.pi / 5)*np.sin(t)
	z = 2*np.random.rand(number_of_samples-1)#np.mod(43*t,1)
	data = np.stack((x,y,z),0).T
	return np.array(data)


def computeGraphEmbedding( graphLaplacian, myrange):
	# Compute dominant eigenvectors
	eigvals, eigvecs = sp.linalg.eigh( l, eigvals=(0,max(myrange)) )
	return eigvecs[:,myrange]


def visualize3DData( data, edata, c):
	fig = plt.figure(figsize=(10,8))
	ax1  = fig.add_subplot(211, projection='3d')
	ax1.scatter( data[:,0], data[:,1], data[:,2], c=c)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_zlabel('z')
	ax1.set_title('Original Data')

	ax2  = fig.add_subplot(212, projection='3d')
	ax2.scatter( edata[:,0], edata[:,1], edata[:,2], c=c)
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_zlabel('z')
	ax2.set_title('Embedded Data')

	plt.show()


# Demonstrate the work flow 
if __name__=='__main__':
	
	# Build data then save it
	data = buildSwissRoll()	
	writeToFile(data, 'swissroll_data.csv');

	# Or read input data
	# data = readFromFile('swissroll_data.csv');

	# Compute the graph Laplacian
	l = buildGraphLaplacian( data, .5, 21)

	# Embed onto the first three nontrivial eigenvectors
	embedded_data = computeGraphEmbedding( l, (1,2,3) )

	# Visualize embedding
	visualize3DData( data, embedded_data, range(0,len(data)) )
