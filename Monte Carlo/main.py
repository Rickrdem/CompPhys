"""
This is the main file of the Monte Carlo simulation of the Ising Model for
the course computational physics at Leiden University.
The program consists of 5 files:
    main.py, 
    observables.py, 
    monte_carlo.py, 
    dynamics.py and
    viewport.py
The program is started by running main.py. 
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""

import matplotlib.pyplot as plt

import observables as obs
from dynamics import Dynamics
from viewer import Viewer


if __name__ == "__main__":
    plt.close("all")
    print('Monte Carlo simulation of the Ising model started')
    
    # Initial values
    J = 1
    T = 2.26
    
    simulation_state = Dynamics(J, T)
    print('Simulation state created')
        
       
    viewport = Viewer(simulation_state)

    plt.show()
        
    print("""
        Pair-pair coupling: J = {j}
        Temperature: T = {t:.2f}
        Magnetic field: H = {h}
        """.format(j=simulation_state.J,
                    t=simulation_state.T,
                    h=simulation_state.magnetic_field))

    obs.magnetization(simulation_state)
    # obs.susceptibility(simulation_state)
    plt.show()
    
    print("Done")