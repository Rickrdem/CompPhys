"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import numba

@numba.njit()
def metropolis(state, neighbours, temp=1, j_coupling=1, magnetic_field=0, steps=100):
    """
    Metropolis algorithm.
    
    :param state (1d array): spin of each point 
    :param neighbours (Nx4 array): contains indices of 4 neighbours of each
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a perturbation
    """
    # rows, columns = state.shape

    # Energy = np.empty(steps)
    for step in range(steps):
        # Energy[step] = np.sum(state)
        # location = (np.random.randint(0,rows), np.random.randint(0,columns))#np.unravel_index(np.random.randint(state.size), state.shape)
        location = np.random.randint(0, len(state))
        delta_E = 0
        for neighbour_index in list(neighbours[location]):
            delta_E += 2*j_coupling*state[location] * state[neighbour_index] + 2*magnetic_field*state[location]
        if delta_E < 0:
            pass
        elif np.random.rand() > np.exp(-1*delta_E/temp):
            continue
        state[location] *= -1
    return state

@numba.njit()
def wolff(state, neighbours, temp=1, j_coupling=1, magnetic_field=0, steps=100):
    """
    Wolff algorithm.
    
    :param state (1d array): spin of each point 
    :param neighbours (Nx4 array): contains indices of 4 neighbours of each
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a perturbation
    """
    # rows, columns = state.shape

    ghost_spin = -1  # we use a ghost spin at index '-1' that we use to model the effect of the external magnetic field

    for step in range(steps):
        location = np.random.randint(0,len(state))  # random site to start

        queue = [location]
        cluster = [location]

        while len(queue) > 0:  # while there are elements left
            newqueue = queue.copy()
            queue.clear()
            for current in newqueue:

                if current < 0:  # the 'ghost spin' that represents the field
                    nearest_neighbours = []
                    for i in range(len(state)):
                        if i not in cluster:
                            nearest_neighbours.append(i)
                else:
                    nearest_neighbours = [-1]
                    for i in neighbours[current]:
                        if i not in cluster:
                            nearest_neighbours.append(i)

                for neighbour in nearest_neighbours:
                    if neighbour not in cluster:
                        if neighbour == -1:  # neighbour is the ghost
                            delta_z = -2*magnetic_field * state[current] * ghost_spin
                        elif current == -1:  # current is the ghost
                            delta_z = -2*magnetic_field * state[neighbour] * ghost_spin
                        else:
                            delta_z = -2*j_coupling * state[neighbour] * state[current]

                        if np.random.rand() < (1-np.exp(delta_z/temp)):
                            queue.append(neighbour)
                            cluster.append(neighbour)
                if current == -1:
                    ghost_spin *= -1
                else:
                    state[current] *= -1
    return state

@numba.njit()
def heat_bath(state, neighbours, temp=1, j_coupling=1, magnetic_field=1, steps=100):
    """
    Heat Bath algorithm.
    
    :param state (1d array): spin of each point (num_spins points)
    :param neighbours (Nx4 array): contains indices of 4 neighbours of each
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a perturbation
    """

    energy_total = 0

    for step in range(steps):
        position = np.random.randint(0, len(state))
        local_field = 0
        for neighbour in neighbours[position]:
            local_field += state[neighbour]

        if np.random.rand() < 1/(1+np.exp(-2*(j_coupling*local_field + magnetic_field)/temp)):
            state[position] = 1
        else:
            state[position] = -1

    return state

@numba.njit(parallel=True)
def checkerboard(state, neighbours, temp=1, j_coupling=1, magnetic_field=1, steps=1):
    """
    Chekerboard algorithm.
    
    :param state (1d array): spin of each point (num_spins points)
    :param neighbours (Nx4 array): contains indices of 4 neighbours of each
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a perturbation
    """

    for step in numba.prange(steps):
        propabilities = np.random.rand(len(state))
        # print(state)
        # even
        for i in numba.prange(len(state)//2):
            index = i*2
            delta_E = 0
            for j in neighbours[index]:
                delta_E += 2 * j_coupling * state[index] * state[j] + 2 * magnetic_field * state[index]
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
                delta_E += 2 * j_coupling * state[index] * state[j] + 2 * magnetic_field * state[index]
            if delta_E < 0:
                pass
            elif propabilities[index] > np.exp(-1 * delta_E / temp):
                continue
            state[index] *= -1
    return state
