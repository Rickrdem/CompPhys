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
@author: Lennard Kwakernaak (1691988)
@author: Rick Rodrigues de Mercado (1687115)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys


import observables as obs
from dynamics import Dynamics
from viewer import Viewer

def progress(count, total, status=''):
    """
    Plot a loading bar while running a headless simulation
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r \r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

if __name__ == "__main__":
    plt.close("all")
    print('Monte Carlo simulation of the Ising model started')
    
    # Initial values
    headless = True
    j_coupling = 1
    temp = 2 / (np.log(1 + np.sqrt(2)))
#    temp = 1
    
    simulation_state = Dynamics(j_coupling, temp, algorithm='Checkerboard')
    print('Simulation state created')
        
    if headless:
        iterations = 1000 + 200
        for i in range(iterations):
            simulation_state.update_and_save()
            progress(i, iterations)
    else: 
        viewport = Viewer(simulation_state)

        
    start = 200
    if len(simulation_state.magnetization) > start:
        simulation_state = obs.trim_data(simulation_state, start)

        mag, mag_err = obs.magnetization(simulation_state)
        sus, sus_err = obs.susceptibility(simulation_state)
        heat, heat_err = obs.specific_heat(simulation_state)

        print(r"""
            Pair-pair coupling: j_coupling = {j}
            Temperature: temp = {temp:.2f}
            Magnetic field: H = {field}
            Average magnetization: <m> {m:.4f} $\pm$ {m_err:.4f}
            Susceptibility: $\chi$ = {s:.4f} $\pm$ {s_err:.4f}
            Specific heat: C_v = {h:.4f} $\pm$ {h_err:.4f}
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