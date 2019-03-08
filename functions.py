import numpy as np
#import numba
from itertools import product


def distance_completely_vectorized(positions, boxsize):
    """
    Calculates the distance vectors r_ij for every particle i to every particle j
    in each dimension

    """
    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]
    L = boxsize[0]
    direction_vector = (positions[:,np.newaxis,:] - positions[np.newaxis,:,:] + L/2+L)%L - L/2
    return direction_vector

def distance_matrix(positions, boxsize):
    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]
    L = boxsize[0]
    distances =  np.sqrt(np.sum(np.square(
                                (positions[:,np.newaxis,:] - positions[np.newaxis,:,:] - L/2)%L + L/2)
                                , axis=2)
                        )
    return distances

def sc_lattice(boxsize, a=1, dim=3):
    """Returns a simple cubic lattice with  lattice constant a .
    Points are  a/4 away from the boundary.
    Returns for dim=3: nparray([[x1,y1,z1],[x2,y2,z2],...,[xN,yN,zN]]), where
    N is the number of nodes.
    """
    return np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))
        

def fcc_lattice(boxsize, a=1, dim=3):
    sc1 = np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))

    sc2 = np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))
    sc2[:,0] += .5*a
    sc2[:,1] += .5*a

    sc3 = np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))
    sc3[:,0] += .5*a
    sc3[:,2] += .5*a

    sc4 = np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))
    sc4[:,1] += .5*a
    sc4[:,2] += .5*a
    
    fcc = np.concatenate((sc1, sc2, sc3, sc4))
    
    mask = np.where(np.all((fcc>=boxsize)==False, axis=1)==False) #remove all positions>L
    fcc = np.delete(fcc, mask, axis=0)
    
#    mask = np.where((fcc[:,0]>=boxsize/2)==False) #remove all positions>L
#    fcc = np.delete(fcc, mask, axis=0) 
    
    return fcc

def U_reduced(r):
    return -4 * (np.divide(1, np.power(r,12), out=np.zeros_like(r), where=r!=0) - np.divide(1, np.power(r,6), out=np.zeros_like(r), where=r!=0))
    
def force_reduced(r):
    """Calculate the vectorised force matrix using the distances between all particles r"""
    dist = abs(r)
    direction = np.divide(r, dist[:,:,np.newaxis], out=np.zeros_like(r), where=r != 0)
    return direction*24*(2*np.power(dist, -13, out=np.zeros_like(dist), where=dist!=0)
                          - np.power(dist, -7, out=np.zeros_like(dist), where=dist!=0)
                          )[:,:,np.newaxis]    
    
def abs(r):
    """Return the magnitude of the vectors contained in r"""
    return np.sqrt(np.sum(np.square(r), axis=-1))

