"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def magnetization(simulation_state):
    """
    Plots the average magnetization of the system. 
    """
    magnetization = simulation_state.magnetization
    time = np.arange(0, len(magnetization), 1)
    
    fig, ax = plt.subplots()
    ax.plot(time, magnetization)
    
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle m \rangle$")
    
    fig.tight_layout()
    fig.show()
    
def susceptibility(simulation_state):
    """
    Plots the susceptibility per spin. 
    """
    susceptibility = np.var(simulation_state.magnetization)*simulation_state.N/simulation_state.T
    print(susceptibility)
    # time = np.arange(0, len(susceptibility), 1)
    #
    # fig, ax = plt.subplots()
    # ax.plot(time, susceptibility)
    #
    # ax.set_xlabel("Time")
    # ax.set_ylabel(r"$\chi$ ($k_BT$)")
    #
    # fig.tight_layout()
    # fig.show()

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