import numpy as np
#import numba
from itertools import product


##@numba.njit(fastmath=True, parallel=True, debug=False)
#def distance(positions, size=(1,1)):
#    """Calculate the distance in 2D between each 2 points according to the 
#    minimum image convention"""
#    
#    shape = len(positions)
#    result = np.empty((shape,shape))
#    for i in range(shape):
#        result[i,i] = 0.0
#        for j in range(i+1,shape):
#            dx = dist(positions[i,0], positions[j,0], size[0])
#            dy = dist(positions[i,1], positions[j,1], size[1])
#            #dz = dist(position[i,2], position[j,2], size[2]) Something 3D with extra for loop may take too long
#
#            result[i,j] = result[j,i] = (dx**2+dy**2)**(0.5)
#        
#    return result
#
#def dist(x1, x2, L):
#    """Calculate the distance in 1D between x1 and x2 according to the minimum 
#    image convention in a box of length L"""
#    dx = min(np.abs(x1 - (x2 + L)),
#             np.abs(x1 - x2), 
#             np.abs(x1 - (x2 - L)))
#
#    return dx
#
#def periodic_distance_vector(positions, boxsize=(1,1)):
#    """
#    Returns the smallest absolute distance given periodic boundaries defined by boxsize in the minimum image convention.
#    :param positions:
#    NxD matrix
#    :param boxsize:
#    Two-tuple containing the dimensions of the periodic bounds L
#    """
#    r = positions[:,0] + 1j*positions[:,1]
#    np.min(np.dstack([
#            np.abs(r-r[:,np.newaxis]),
#            np.abs(r-r[:,np.newaxis] - boxsize[0]),
#            np.abs(r - r[:, np.newaxis] + boxsize[0]),
#            np.abs(r - r[:, np.newaxis] - 1j*boxsize[1]),
#            np.abs(r - r[:, np.newaxis] + 1j*boxsize[1])]
#        ), axis=2)
#
#def distance_vector(positions):
#    """
#    Vectorized distance funtion for a
#    :param positions:
#    :return:
#    """
#    x = positions[:,0]
#    y = positions[:,1]
#    dx = x[:,np.newaxis]-x[:,np.newaxis].T
#    dy = y[:,np.newaxis]-y[:,np.newaxis].T
#    return np.sqrt(np.square(dx)+np.square(dy))


def distance_completely_vectorized(positions, boxsize):
    """
    Calculates the distance vectors r_ij for every particle i to every particle j
    in each dimension

    """
    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]
    L=boxsize[0]
    direction_vector =  (positions[:,np.newaxis,:] - positions[np.newaxis,:,:] - L/2)%L-L/2 # N,N,di
    return direction_vector

def distance_matrix(positions, boxsize):
    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]
    L = boxsize[0]
    distances =  np.sqrt(np.sum(np.square(
                                (positions[:,np.newaxis,:] - positions[np.newaxis,:,:] - L/2)%L-L/2)
                                , axis=2)
                        )
    return distances

#def generate_lattice(particles_per_axis, boxsize):
#    return np.mgrid.__getitem__([
#        slice(1,i-1,particles_per_axis*1j) for i in boxsize
#    ])

def sc_lattice(boxsize, a=1, dim=3):
    """Returns a simple cubic lattice with  lattice constant a .
    Points are  0.5*a away from the boundary.
    Returns for dim=3: nparray([[x1,y1,z1],[x2,y2,z2],...,[xN,yN,zN]]), where
    N is the number of nodes.
    """
    return np.array(list(product(np.arange(a/2, boxsize, a), repeat=dim)))
        

def fcc_lattice(boxsize, a=1, dim=3):
    sc1 = np.array(list(product(np.arange(a/2, boxsize, a), repeat=dim)))

    sc2 = np.array(list(product(np.arange(a/2, boxsize, a), repeat=dim)))
    sc2[:,0] += .5
    sc2[:,1] += .5

    sc3 = np.array(list(product(np.arange(a/2, boxsize, a), repeat=dim)))
    sc3[:,0] += .5
    sc3[:,2] += .5

    sc4 = np.array(list(product(np.arange(a/2, boxsize, a), repeat=dim)))
    sc4[:,1] += .5
    sc4[:,2] += .5
    return np.concatenate((sc1, sc2, sc3, sc4))


#def U(r,  sigma=119.8, epsilon=3.405):
#    return 4 * epsilon * (sigma**12 /np.power(r, 12) - sigma**6 / np.power(r, 6))

def U_reduced(r):
    return 4 * (np.divide(1, np.power(r,12), out=np.zeros_like(r), where=r!=0) - np.divide(1, np.power(r,6), out=np.zeros_like(r), where=r!=0)) 

#def absolute_force(r,  sigma=119.8, epsilon=3.405):
#    """
#    F = - 1/r * dU/dr * vec{x}
#    """
#    return  ( 24 * (sigma ** 6) * epsilon * np.divide(np.power(r,6) - 2*sigma**6, np.power(r, 14), out=np.zeros_like(r), where=r!=0))

def absolute_force_reduced(r):
    """
    F = - 1/r * dU/dr * vec{x}
    """
    return  ( 24 *( (np.divide(2, np.power(r,14), out=np.zeros_like(r), where=r!=0)) - (np.divide(1, np.power(r,8), out=np.zeros_like(r), where=r!=0))))
    
def force_reduced(r):
    """Calculate the vectorised force matrix using the distances between all particles r"""
    return (24 * ((np.divide(2, np.power(r, 13), out=np.zeros_like(r), where=r != 0)) - (
        np.divide(1, np.power(r, 7), out=np.zeros_like(r), where=r != 0))))
    
    
    