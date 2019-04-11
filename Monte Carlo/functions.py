import numpy as np
from scipy import ndimage as ndi

def combination(length, digit):
    return np.arange(length)//(3**digit)%3-1

def window(data, location):
    """
    Return a windowed view of the data at location assuming periodic boundary conditions.
    :param data: The original dataset to slice the view out of.
    :param location: The index at which to view the window
    :return: A 3x3 window around location with periodic boundary conditions.
    """
    x, y = location
    width, height = data.shape
    xrange = [x-1, x, x+1]
    yrange = [y-1, y, y+1]
    indices = [k + l*height for k in xrange for l in yrange]
    return data.take(indices, mode='wrap').reshape(3,3)

def nn_interaction(shape):
    middle = len(shape)//2
    m = shape[middle]
    shape[middle] = 0
    return np.sum(m*shape)

def H(state, J):
    """ Hamiltonian operator on a state vector for free bounds"""
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    return -J * np.sum(ndi.generic_filter(
        state, function=nn_interaction, footprint=footprint, mode='wrap'))

def decide(dE, T):
    """ Decide on whether to keep the perturbation or not"""
    if dE < 0: # catch easy cases
        return True
    chance = np.exp(-dE/T) #
    return np.random.choice((True, False), p=(chance, 1-chance))

def perturb(state, J, T):
    """ Perturb the state when the energy is lowered or
    keep it with a chance defined by the acceptance probability
    See wikipedia on monte carlo ising simulation"""
    location = np.unravel_index(np.random.randint(state.size), state.shape)
    microstate = window(state, location)
    E = H(microstate, J)
    microstate[tuple(d//2 for d in microstate.shape)] *= -1
    dE = H(microstate, J) - E
    
    if decide(dE, T):
        state[location] *= -1
        return state    
    # otherwise
    return state