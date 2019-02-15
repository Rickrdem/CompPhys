import numpy as np
#import numba
from itertools import permutations


#@numba.njit(fastmath=True, parallel=True, debug=False)
def distance(positions, size=(1,1)):
    """Calculate the distance in 2D between each 2 points according to the 
    minimum image convention"""
    
    shape = len(positions)
    result = np.empty((shape,shape))
    for i in range(shape):
        result[i,i] = 0.0
        for j in range(i+1,shape):
            dx = dist(positions[i,0], positions[j,0], size[0])
            dy = dist(positions[i,1], positions[j,1], size[1])
            #dz = dist(position[i,2], position[j,2], size[2]) Something 3D with extra for loop may take too long

            result[i,j] = result[j,i] = (dx**2+dy**2)**(0.5)
        
    return result

def dist(x1, x2, L):
    """Calculate the distance in 1D between x1 and x2 according to the minimum 
    image convention in a box of length L"""
    dx = min(np.abs(x1 - (x2 + L)),
             np.abs(x1 - x2), 
             np.abs(x1 - (x2 - L)))

    return dx

def periodic_distance_vector(positions, boxsize=(1,1)):
    """
    Returns the smallest absolute distance given periodic boundaries defined by boxsize in the minimum image convention.
    :param positions:
    NxD matrix
    :param boxsize:
    Two-tuple containing the dimensions of the periodic bounds L
    """
    r = positions[:,0] + 1j*positions[:,1]
    np.min(np.dstack([
            np.abs(r-r[:,np.newaxis]),
            np.abs(r-r[:,np.newaxis] - boxsize[0]),
            np.abs(r - r[:, np.newaxis] + boxsize[0]),
            np.abs(r - r[:, np.newaxis] - 1j*boxsize[1]),
            np.abs(r - r[:, np.newaxis] + 1j*boxsize[1])]
        ), axis=2)

def distance_vector(positions):
    """
    Vectorized distance funtion for a
    :param positions:
    :return:
    """
    x = positions[:,0]
    y = positions[:,1]
    dx = x[:,np.newaxis]-x[:,np.newaxis].T
    dy = y[:,np.newaxis]-y[:,np.newaxis].T
    return np.sqrt(np.square(dx)+np.square(dy))


def distance_completely_vectorized(positions, boxsize):
    """
    Calculates the distance vectors r_ij for every particle i to every particle j

    :param positions:
    :param boxsize:
    :param dimensions:
    :return:
    """
    dimensions = len(boxsize)

    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]

    monstermatrix =  (positions[:,np.newaxis,:] - positions[np.newaxis,:,:] - L/2)%L-L/2 # N,N,di
    return monstermatrix

def generate_lattice(particles_per_axis, dimensions):
    return np.mgrid.__getitem__([
        slice(-i,i,particles_per_axis*1j) for i in boxsize
    ])
        
        
def U(r, sigma, epsilon):
    return 4 * epsilon * (sigma**12 /np.power(r, 12) - sigma**6 / np.power(r, 6))


def F(direction, r, sigma, epsilon):
    """
    F = 1/r * dU/dr * vec{x}
    """
    return - 24 * (sigma ** 6) * epsilon * np.divide(np.power(r,6) - 2*sigma**6, np.power(r, 14)) * direction