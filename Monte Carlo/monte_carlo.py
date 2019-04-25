"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import numba
#from scipy import ndimage as ndi


@numba.njit()
def metropolis(state, neighbours, temp=1, J=1, H=0, steps=100):
    """
    Metrpolis algorithm.
    
    :param state (1d array): spin of each point 
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param J (float): Spin-spin coupling.
    :param H (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
    """
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
def wolff(state, neighbours, temp=1, J=1, H=0, steps=100):
    """
    Wolff algorithm.
    
    :param state (1d array): spin of each point 
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param J (float): Spin-spin coupling.
    :param H (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
    """
   
    
    # rows, columns = state.shape
    propability = 1. - np.exp(-2*J/temp)
    propability_field = 1 - np.exp(-2*np.abs(H)/temp)
    # energy = np.empty(steps)
    field_spin = np.sign(H)

    for step in range(steps):
        # energy[step] = np.sum(state)
        # location = (np.random.randint(0,rows), np.random.randint(0,columns))
        location = np.random.randint(0,len(state))  # random site to start

        queue = [location]
        cluster = [location]

        while len(queue) > 0:  # while there are elements left
            current = np.random.choice(np.array(queue))

            if current < 0:  # the 'ghost spin' that represents the field
                nearest_neighbours = []
                for i in state:
                    nearest_neighbours.append(i)
            else:
                nearest_neighbours = [-1]
                for i in neighbours[current]:
                    nearest_neighbours.append(i)
                # nearest_neighbours.append(-1)

            for neighbour in nearest_neighbours:
                if neighbour not in cluster:
                    if neighbour > 0:
                        if state[current] == state[neighbour] and np.random.rand() < propability:
                            queue.append(neighbour)
                            cluster.append(neighbour)
                    else:  # external field
                        if state[current]*field_spin*np.sign(H) > 0 and np.random.rand() < propability_field:
                            queue.append(neighbour)
                            cluster.append(neighbour)
            queue.remove(current)
        for current in cluster:
            if current > 0:
                state[current] *= -1
            else:
                field_spin *= -1
    return state

@numba.njit()
def heat_bath(state, neighbours, temp=1, J=1, H=1, steps=100):
    """
    Heat Bath algorithm.
    
    :param state (1d array): spin of each point (N points)
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param J (float): Spin-spin coupling.
    :param H (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
    """

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

@numba.njit(parallel=True)
def checkerboard(state, neighbours, temp=1, J=1, H=1, steps=1):
    """
    Chekerboard algorithm.
    
    :param state (1d array): spin of each point (N points)
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param J (float): Spin-spin coupling.
    :param H (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
    """

    for step in numba.prange(steps):
        propabilities = np.random.rand(len(state))
        # print(state)
        # even
        for i in range(len(state)//2):
            index = i*2
            delta_E = 0
            for j in neighbours[index]:
                delta_E += 2 * J * state[index] * state[j] + 2 * H * state[index]
            if delta_E < 0:
                pass
            elif propabilities[index] > np.exp(-1 * delta_E / temp):
                continue
            state[index] *= -1

        # odd
        for i in numba.prange(len(state)//2):
            index = 2*i+1
            delta_E = 0
            for j in neighbours[index]:
                delta_E += 2 * J * state[index] * state[j] + 2 * H * state[index]
            if delta_E < 0:
                pass
            elif propabilities[index] > np.exp(-1 * delta_E / temp):
                continue
            state[index] *= -1

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