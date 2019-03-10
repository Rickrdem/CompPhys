import numpy as np
import numba
from itertools import product


#def distance_completely_vectorized(positions, boxsize):
#    """
#    Calculates the distance vectors r_ij for every particle i to every particle j
#    in each dimension
#
#    """
#    L = np.asarray(boxsize)[np.newaxis,np.newaxis,:]
#    L = boxsize[0]
#    direction_vector = np.mod((positions[:,np.newaxis,:] - positions[np.newaxis,:,:] + L/2),L) - L/2
#    return direction_vector
#
#@numba.njit()
#def mindist(a,b,L):
#    return (a-b+L/2)%L - L/2

@numba.jit(parallel=True)
def distance_jit(positions, boxsize):
    L = np.array(boxsize)
    particles, dimensions = positions.shape
    distances = np.empty((particles,particles,dimensions), dtype=positions.dtype)
    for i in range(particles):
        for k in range(dimensions):
            distances[i, i, k] = 0
            for j in range(i+1,particles):
                # d = ((positions[i,k] - positions[j,k] + L[k]/2) % L[k]) - L[k]/2
                d = positions[i,k] - positions[j,k]
                if d<-L[k]/2.:
                    d = d+L[k]
                elif d>L[k]/2.:
                    d = d-L[k]
                distances[i,j,k] = d
                distances[j,i,k] = -d
    return distances

#def sc_lattice(boxsize, a=1, dim=3):
#    """Returns a simple cubic lattice with  lattice constant a .
#    Points are  a/4 away from the boundary.
#    Returns for dim=3: nparray([[x1,y1,z1],[x2,y2,z2],...,[xN,yN,zN]]), where
#    N is the number of nodes.
#    """
#    return np.array(list(product(np.arange(a/4, boxsize-a/4, a), repeat=dim)))
        

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
        
    return fcc

def U_reduced(r):
    return 4 * (np.divide(1, np.power(r,12), out=np.zeros_like(r), where=r!=0) - np.divide(1, np.power(r,6), out=np.zeros_like(r), where=r!=0))

@numba.njit()
def sum_potential_jit(r):
    """
    Calculate the total potential of the whole system in the standard Lennard-Jones potential
    :param r: NxNxd matrix containing all the distances between all particles in reduced units.
    :return: float of the total potential.
    """
    particles = r.shape[0]
    dimensions = r.shape[2]
    U = np.empty((particles, particles, dimensions))
    Total_U = 0
    for i in range(particles):
        for k in range(dimensions):
            U[i, i, k] = 0
        for j in range(1, particles):
            m = 0  # magnitude of the vector

            for k in range(dimensions):
                m += r[i, j, k] * r[i, j, k]
            m = np.sqrt(m)

            for k in range(dimensions):
                if m == 0:
                    U[i, j, k] = 0
                else:
                    U[i, j, k] =  4 * (
                                1 / (m * m * m * m * m * m * m * m * m * m * m * m)  # absolutely ridiculous performance increase
                                - 1 / (m * m * m * m * m * m))
                U[j, i, k] = -U[i, j, k]
                Total_U += U[i, j, k]
    return Total_U

#def absolute_force_reduced(r):
#    """
#    F = - 1/r * dU/dr * vec{x}
#    """
#    return  ( 24 *( (np.divide(2, np.power(r,14), out=np.zeros_like(r), where=r!=0)) - (np.divide(2, np.power(r,8), out=np.zeros_like(r), where=r!=0))))

#def force_reduced(r):
#    """Calculate the vectorised force matrix using the distances between all particles r"""
#    dist = abs(r)
#    direction = np.divide(r, dist[:,:,np.newaxis], out=np.zeros_like(r), where=r != 0)
#    return direction*24*(2*np.power(dist, -13, out=np.zeros_like(dist), where=dist!=0)
#                          - np.power(dist, -7, out=np.zeros_like(dist), where=dist!=0)
#                          )[:,:,np.newaxis]
#
#def force_fast(r):
#    """
#    Apparently r*r*r is faster than np.power(r,3). Who knew!
#    :param r: NxNxd matrix containing all the distances between all particles in reduced units
#    """
#    dist = abs(r)
#    direction = np.divide(r,dist[:,:,None], where=r!=0, out=np.zeros_like(r))
#    out = np.zeros_like(r)
#    return direction * 24 *(np.divide(2,dist*dist*dist*dist*dist*dist*dist*dist*dist*dist*dist*dist*dist, out=np.zeros_like(dist), where=dist!=0)
#                            - np.divide(1,dist*dist*dist*dist*dist*dist*dist, out=np.zeros_like(dist), where=dist!=0))[:,:,None]
#
@numba.njit()
def force_jit(r):
    """
    Calculate the total force on particle N due to the contributions of all other particles in the standard Lennard-Jones potential
    :param r: NxNxd matrix containing all the distances between all particles in reduced units.
    :return: Nxd matrix containing the force on every particle N in every dimension d.
    """
    particles = r.shape[0]
    dimensions = r.shape[2]
    force = np.empty((particles, particles, dimensions))
    for i in range(particles):
        for k in range(dimensions):
            force[i, i, k] = 0
        for j in range(i + 1, particles):
            square = 0  # magnitude of the vector squared

            for k in range(dimensions):
                square += r[i, j, k] * r[i, j, k]

            for k in range(dimensions):
                if square == 0:
                    force[i, j, k] = 0
                else:
                    sixth = square ** 3
                    twelvth = sixth ** 2
                    force[i, j, k] = r[i, j, k] * 24 * (2 / (twelvth * square) - 1 / (sixth * square))
                force[j, i, k] = -force[i, j, k]
    return force
    
def abs(r):
    """Return the magnitude of the vectors contained in r"""
    return np.sqrt(np.sum(np.square(r), axis=-1))

