import numpy as np
import numba
from scipy import ndimage as ndi


# @numba.njit()
def metropolis(state, neighbours, temp=1, J=1, H=0, steps=100):
    # rows, columns = state.shape

    # Energy = np.empty(steps)
    for step in range(steps):
        # Energy[step] = np.sum(state)
        # location = (np.random.randint(0,rows), np.random.randint(0,columns))#np.unravel_index(np.random.randint(state.size), state.shape)
        location = np.random.randint(0, len(state))
        delta_E = 0
        for neighbour_index in list(neighbours[location]):
            delta_E += 2*J*state[location] * state[neighbour_index] + 2*H*state[location]
        if delta_E < 0:
            pass
        elif np.random.rand() > np.exp(-1*delta_E/temp):
            continue
        state[location] *= -1
    return state

@numba.njit()
def wolff(state, neighbours, temp=1, H=0, J=1, steps=1):
    # rows, columns = state.shape

    propability = 1. - np.exp(-2*J/temp)
    propability_field = 1 - np.exp(-2*J*np.abs(H)/temp)
    # energy = np.empty(steps)

    for step in range(steps):
        # energy[step] = np.sum(state)
        # location = (np.random.randint(0,rows), np.random.randint(0,columns))
        location = np.random.randint(0,len(state))

        queue = [location]
        cluster = [location]

        while len(queue) > 0:  # while there are elements left
            i = np.random.choice(np.array(queue))
            for j in list(neighbours[i]):
                if j not in cluster:
                    if (state[i]==state[j] and np.random.rand() < propability)\
                            or (state[i]*state[j]*np.sign(H) > 0 and np.random.rand() < propability_field):
                        queue.append(j)
                        cluster.append(j)
            queue.remove(i)
        for i in cluster:
            state[i] *= -1
    return state

@numba.njit()
def heat_bath(state, neighbours, temp=1, H=1, J=1, steps=1):
    energy_total = 0

    for step in range(steps):
        position = np.random.randint(0, len(state))
        local_field = 0
        for neighbour in neighbours[position]:
            local_field += state[neighbour]

        if np.random.rand() < 1/(1+np.exp(-2*(J*local_field + H)/temp)):
            state[position] = 1
        else:
            state[position] = -1

    return state

# def combination(length, digit):
#     return np.arange(length)//(3**digit)%3-1
#
# def window(data, location):
#     """
#     Return a windowed view of the data at location assuming periodic boundary conditions.
#     :param data: The original dataset to slice the view out of.
#     :param location: The index at which to view the window
#     :return: A 3x3 window around location with periodic boundary conditions.
#     """
#     x, y = location
#     width, height = data.shape
#     xrange = [x-1, x, x+1]
#     yrange = [y-1, y, y+1]
#     indices = [k + l*height for k in xrange for l in yrange]
#     return data.take(indices, mode='wrap').reshape(3,3)
#
# def nn_interaction(shape):
#     middle = len(shape)//2
#     m = shape[middle]
#     shape[middle] = 0
#     return np.sum(m*shape)
#
# def H(state, J):
#     """ Hamiltonian operator on a state vector for free bounds"""
#     footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
#     return -J * np.sum(ndi.generic_filter(
#         state, function=nn_interaction, footprint=footprint, mode='wrap'))
#
# def decide(dE, T):
#     """ Decide on whether to keep the perturbation or not"""
#     if dE < 0: # catch easy cases
#         return True
#     chance = np.exp(-dE/T) #
#     return np.random.choice((True, False), p=(chance, 1-chance))
#
# def perturb(state, J, T):
#     """ Perturb the state when the energy is lowered or
#     keep it with a chance defined by the acceptance probability
#     See wikipedia on monte carlo ising simulation"""
#     location = np.unravel_index(np.random.randint(state.size), state.shape)
#     microstate = window(state, location)
#     E = H(microstate, J)
#     microstate[tuple(d//2 for d in microstate.shape)] *= -1
#     dE = H(microstate, J) - E
#
#     if decide(dE, T):
#         state[location] *= -1
#         return state
#     # otherwise
#     return state