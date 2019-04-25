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

import numpy as np
import matplotlib.pyplot as plt

import observables as obs
from dynamics import Dynamics
from viewer import Viewer


if __name__ == "__main__":
    plt.close("all")
    print('Monte Carlo simulation of the Ising model started')
    
    # Initial values
    j_coupling = 1
    temp = 2 / (np.log(1 + np.sqrt(2)))
    
    simulation_state = Dynamics(j_coupling, temp)
    print('Simulation state created')
        
       
    viewport = Viewer(simulation_state)

    plt.show()

    start = 0
    if len(simulation_state.magnetization) > start:
        simulation_state = obs.trim_data(simulation_state, start)

        mag, mag_err = obs.magnetization(simulation_state)
        sus, sus_err = obs.susceptibility(simulation_state)
        heat, heat_err = obs.specific_heat(simulation_state)

        print(r"""
            Pair-pair coupling: j_coupling = {j}
            Temperature: temp = {temp:.2f}
            Magnetic field: H = {field}
            Average magnetization: <m> {m:.3f} $\pm$ {m_err:.3f}
            Susceptibility: $\chi$ = {s:.2f} $\pm$ {s_err:.2f}
            Specific heat: C_v = {h:.2f} $\pm$ {h_err:.2f}
            """.format(j=simulation_state.j_coupling,
                        temp=simulation_state.temp,
                        field=simulation_state.magnetic_field,
                        m=mag,
                        m_err=mag_err,
                        s=sus,
                        s_err=sus_err,
                        h=heat,
                        h_err=heat_err
                       ))
    else:
        print('Equilibrium was not yet reached. Please try again...')

    plt.show()
    
    print("Done")