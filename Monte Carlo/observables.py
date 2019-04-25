"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def trim_data(simulation_state, start, end=None):
    """
    Trim the data to remove a set of time steps.
    :param simulation_state: The state to trim in time
    :param start: The number of states to trim off of the front of the dataset.
    :param end: The number of states to trim off of the back of the dataset.
    :return: Trimmed simulation state.
    """
    simulation_state.energy = simulation_state.energy[start:end]
    simulation_state.magnetization = simulation_state.magnetization[start:end]
    return simulation_state

def magnetization(simulation_state):
    """
    Plots the average magnetization of the system. 
    """
    magnetization = simulation_state.magnetization
    average_magnetization = np.average(magnetization)
    magnetization_error = bootstrap(magnetization, np.average)
    time = np.arange(0, len(magnetization), 1)
    
    fig, ax = plt.subplots()
    ax.plot(time, magnetization)
    
    ax.set_xlabel("Time steps")
    ax.set_ylabel("m")
    
    fig.tight_layout()
    fig.show()
    return average_magnetization, magnetization_error
    
def susceptibility(simulation_state):
    """
    Calculates the susceptibility per spin.
    """
    func = lambda x: np.var(x)*simulation_state.N/simulation_state.T
    sus = func(simulation_state.magnetization)
    sus_error = bootstrap(simulation_state.magnetization, func)
    return sus, sus_error

def specific_heat(simulation_state):
    """
    Calculates the specific heat of the system.
    """
    func = lambda x: np.var(x)/simulation_state.N/(simulation_state.T**2)
    s_heat = func(simulation_state.energy)
    s_heat_err = bootstrap(simulation_state.energy, func)

    return s_heat, s_heat_err

def nn_interaction(shape):
    middle = len(shape)//2
    m = shape[middle]
    shape[middle] = 0
    return np.sum(m*shape)

def energy(state, J=1, H=0):
    """ Hamiltonian operator on a state vector for free bounds"""
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    return -J * np.sum(ndi.generic_filter(
        state, function=nn_interaction, footprint=footprint, mode='wrap'))

def bootstrap(data, function, n=100):
    """Bootstrap an observable, given a specific function"""
    observables = []
    for i in range(n):
        resampled = np.random.choice(data, size=len(data), replace=True)
        observables.append(function(resampled))
    return np.std(observables)

