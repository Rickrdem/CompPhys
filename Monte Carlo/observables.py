import numpy as np
import matplotlib.pyplot as plt

def magnetization(simulation_state):
    magnetization = simulation_state.magnetization
    time = np.arange(0, len(magnetization), 1)
    
    fig, ax = plt.subplots()
    ax.plot(time, magnetization)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnetization")
    
    fig.tight_layout()
    fig.show()
    
def susceptibility(simulation_state):
    susceptibility = simulation_state.susceptibility
    time = np.arange(0, len(susceptibility), 1)
    
    fig, ax = plt.subplots()
    ax.plot(time, susceptibility)
    
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Susceptibility ($k_BT$)")    
    
    fig.tight_layout()
    fig.show()