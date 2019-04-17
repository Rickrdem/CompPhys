import numpy as np
import numba
from scipy import ndimage as ndi

# @numba.njit()
def metropolis(state, neighbours, temp=1, H=0, steps=100):
    rows, columns = state.shape
    Energy = np.empty(steps)
    for step in range(steps):
        Energy[step] = np.sum(state)
        location = (np.random.randint(0,rows), np.random.randint(0,columns))#np.unravel_index(np.random.randint(state.size), state.shape)
        delta_E = 0
        for neighbour in list(neighbours[location]):
            delta_E += 2*state[location] * state[neighbour[0], neighbour[1]] + 2*H*state[location]
        if delta_E < 0:
            pass
        elif np.random.rand() > np.exp(-1*delta_E/temp):
            continue
        state[location] *= -1
    return state, Energy

def wolff(state, neighbours, temp=1, steps=100):
    rows, columns = state.shape
    return state


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