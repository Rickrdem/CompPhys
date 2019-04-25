"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import numba

@numba.njit()
def metropolis(state, neighbours, temp=1, j_coupling=1, magnetic_field=0, steps=100):
    """
    Metrpolis algorithm.
    
    :param state (1d array): spin of each point 
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
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
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
    """
    # rows, columns = state.shape
    propability = 1. - np.exp(-2*j_coupling/temp)
    propability_field = 1 - np.exp(-2*np.abs(magnetic_field)/temp)
    print(propability,propability_field)
    # energy = np.empty(steps)
    field_spin = np.sign(magnetic_field)

    for step in range(steps):
        # energy[step] = np.sum(state)
        # location = (np.random.randint(0,rows), np.random.randint(0,columns))
        location = np.random.randint(0,len(state))  # random site to start

        queue = [location]
        cluster = [location]

        while len(queue) > 0:  # while there are elements left
            # current = np.random.choice(np.array(queue))
            current = queue.pop()


            if current < 0:  # the 'ghost spin' that represents the field
                nearest_neighbours = []
                for i in state:
                    nearest_neighbours.append(i)
                # print('Current value is the external spin')
            else:
                nearest_neighbours = [-1]
                for i in neighbours[current]:
                    nearest_neighbours.append(i)
                # nearest_neighbours.append(-1)
                # print('Current value in array')

            for neighbour in nearest_neighbours:
                if neighbour not in cluster:
                    if neighbour > 0 and current > 0:
                        if state[current] * state[neighbour]==1*np.sign(j_coupling) and np.random.rand() < propability:
                            queue.append(neighbour)
                            cluster.append(neighbour)
                  # current or neighbour is in the external field
                    elif neighbour < 0:
                        if state[current]*field_spin*np.sign(magnetic_field)>0 and np.random.rand() < propability_field:
                            queue.append(neighbour)
                            cluster.append(neighbour)
                            print('added current')
                    elif current < 0:
                        if state[neighbour]*field_spin*np.sign(magnetic_field)>0 and np.random.rand() < propability_field:
                            queue.append(neighbour)
                            cluster.append(neighbour)
                            print('added neighbour')
                        print(state[neighbour]*field_spin*np.sign(magnetic_field))
            # queue.remove(current)
        toflip = cluster
        print(toflip)
        for current in toflip:
            if current > 0:
                state[current] *= -1
            else:
                field_spin *= -1
    return state

@numba.njit()
def heat_bath(state, neighbours, temp=1, j_coupling=1, magnetic_field=1, steps=100):
    """
    Heat Bath algorithm.
    
    :param state (1d array): spin of each point (num_spins points)
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
    :param steps (int): number of iterations.
    
    return: Final state after a pertubation
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
    :param neighbours (Nx4 array): contains indeces of 4 neigbours of each 
                                    spin point
    :param temp (float): Temperature.
    :param j_coupling (float): Spin-spin coupling.
    :param magnetic_field (float): External magnetic field
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