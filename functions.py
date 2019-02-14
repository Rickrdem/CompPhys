import numpy as np
#import numba


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