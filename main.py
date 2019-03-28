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
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""

import numpy as np
import functions as func
import matplotlib.pyplot as plt
import pyglet

from dynamics import Dynamics
from viewport import Viewport, setup
import observables as obs

def main(temperature=0.5, density=1.2, particles=256, starting_state=None, plotting=False):
    M = int(np.round(np.power(particles/4, 1/3.)))
    particles = 4*(M**3) # Number of particles that fit in a fcc
    if starting_state is None:
        L = np.power(particles/density, 1/3)
        lattice_constant = L*np.power(4/particles, 1/3)
        state = func.fcc_lattice(L, a=lattice_constant)
    else:
        state = starting_state
        particles = starting_state.shape[0]
    
    simulation_state = Dynamics(state, T=temperature, size=(L, L, L), dtype=np.float64)

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
    
    start = 60
    if len(simulation_state.kinetic_energy) <= start:
        print("Equilibrium was not reached yet, try again.")
        return
    
    elif plotting:
        # In the first couple of iterations the system is equilibriating.
        simulation_state = obs.trim_data(simulation_state, start)  # remove the first 300 time steps see Verlet et.al.
        obs.velocity_distribution(simulation_state)
        obs.energy(simulation_state)
        obs.diffusion(simulation_state)
        obs.pair_correlation(simulation_state, fig_combined_pc, ax_combined_pc)
        obs.temperature(simulation_state)
        plt.show()
    C_V, std_C_V = obs.specific_heat(simulation_state)
    print("""
    Specific heat {c_v:.2f} +- {std:.4f}
    """.format(c_v=C_V, std= std_C_V))

    
if __name__ == '__main__':
    plt.close('all')
    fig_combined_pc, ax_combined_pc = plt.subplots() # Used to plot (multiple) PCF's

    main(temperature=0.5, density=1.2, particles=800, plotting=True)
    
    """Excersise"""
#    main(temperature=0.5, density=1.2, particles=864, plotting=True)
#    main(temperature=1, density=0.8, particles=864, plotting=True)
#    main(temperature=3, density=0.3, particles=864, plotting=True)
    print('------------------------------------------')
    print("Done!")