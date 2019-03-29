"""
This is the main file of the molucular dynamics simulation of argon for
the course computational physics at Leiden University.
The program consists of 5 files:
    main.py, 
    observables.py, 
    functions.py, 
    dynamics.py and
    viewport.py
and requires two images:
    icon_bits16.png
    icon_bits32.png
The program is started by running main.py. 
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""

import numpy as np
import functions as func
import matplotlib.pyplot as plt
import pyglet
import sys

from dynamics import Dynamics
from viewport import Viewport, setup
import observables as obs


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

def main(temperature=0.5, density=1.2, particles=256, starting_state=None, plotting=False, headless=False):
    """
    :param temperature (float): Temperature of the system
    :param density (float): Density of the system
    :param particles (int): number of particles in the box
    :param starting_state gives (NxM array): specific state, default is fcc
    :param plotting (bool): the turn on/off the plots
    :param headless (bool): run without visualization    
    """
    M = int(np.round(np.power(particles/4, 1/3.)))
    particles = 4*(M**3) # Mach number of particles to fcc
    if starting_state is None:
        L = np.power(particles/density, 1/3)
        lattice_constant = L*np.power(4/particles, 1/3)
        state = func.fcc_lattice(L, a=lattice_constant)
    else:
        state = starting_state
        particles = starting_state.shape[0]
    
    simulation_state = Dynamics(state, T=temperature, size=(L, L, L), dtype=np.float64)

    if not headless:
        print('Simulation state created')
        window = Viewport(simulation_state, drawevery=1)

        print('Windows created')

        pyglet.clock.schedule(simulation_state.update)
        print('Starting app')

        print('------------------------------------------')
        print("""
            Particles {particles}
            Boxsize {boxsize:.2f}
            Temperature {temp:.2f}
            Density {dens:.2f}
            """.format(particles=simulation_state.particles,
                       boxsize=simulation_state.size[0],
                       temp=simulation_state.T,
                       dens=simulation_state.density))

        setup()
        pyglet.app.run()

        pyglet.clock.unschedule(simulation_state.update)
    else:
        print('------------------------------------------')
        print("Booting headless")
        print("""
            Particles {particles}
            Boxsize {boxsize:.2f}
            Temperature {temp:.2f}
            Density {dens:.2f}
            """.format(particles=simulation_state.particles,
                       boxsize=simulation_state.size[0],
                       temp=simulation_state.T,
                       dens=simulation_state.density))

        iterations = 300+1200  # same configuration as in the Verlet paper
        for i in range(iterations):
            simulation_state.update(None)
            progress(i, iterations)
        print("Done")
    
    start = 300
    if len(simulation_state.kinetic_energy) <= start:
        print("Equilibrium was not reached yet, try again.")
        plt.close("all")
        return
    
    elif plotting:
        # In the first couple of iterations the system is equilibriating.
        # remove the first 300 time steps see Verlet et.al.
        simulation_state = obs.trim_data(simulation_state, start)  
        
        obs.plot_velocity_distribution(simulation_state)
        obs.plot_energy(simulation_state)
        obs.plot_diffusion(simulation_state)
        obs.plot_pair_correlation(simulation_state, fig_combined_pc, ax_combined_pc)
        obs.plot_temperature(simulation_state)
        plt.show()
    
    C_V , err_C_V = obs.specific_heat(simulation_state)
    print("""
            Specific heat {c_v:.5f} +- {c_v_err:.5f}
            """.format(
                c_v=C_V,
                c_v_err=err_C_V))
    
if __name__ == '__main__':
    plt.close('all')
    fig_combined_pc, ax_combined_pc = plt.subplots() # Used to plot PCF

    main(temperature=1, density=.8, particles=864, plotting=True, headless=False)

    
    """Excersise"""
#    main(temperature=0.5, density=1.2, particles=864, plotting=True, headless=True)
#    main(temperature=1, density=0.8, particles=864, plotting=True, headless=True)
#    main(temperature=3, density=0.3, particles=864, plotting=True, headless=True)
    print('------------------------------------------')
    print("Done!")