"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import matplotlib.pyplot as plt

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
    susceptibility = simulation_state.susceptibility
    time = np.arange(0, len(susceptibility), 1)
    
    fig, ax = plt.subplots()
    ax.plot(time, susceptibility)
    
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\chi$ ($k_BT$)")    
    
    fig.tight_layout()
    fig.show()