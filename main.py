import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup
import observables as obs
import matplotlib.pyplot as plt

def main(temperature=0.5, density=1.2, particles=256, starting_state=None, plotting=False):
    if starting_state == None:
        L = np.power(particles/density, 1/3)
        lattice_constant = L*np.power(4/particles, 1/3)
        state = func.fcc_lattice(L, a=lattice_constant)
    else:
        state = starting_state
        particles = starting_state.shape[0]

    game = Gamestate(state, T=temperature, size=(L,L,L), dtype=np.float64)

    print('Game created')
    window = Viewport(game, drawevery=1)
    print('Windows created')

    
    window.set_caption('Hexargon')

    pyglet.clock.schedule(game.update)
    print('Starting app')
    
    print('------------------------------------------')   
    print("""
        Particles {particles}
        Boxsize {boxsize:.2f}
        Temperature {temp:.2f}
        Density {dens:.2f}
        Pressure {pressure:.2f}
        """.format(particles=game.particles,
                   boxsize=game.size[0],
                   temp=game.T,
                   dens=game.density,
                   pressure=game.pressure))
    print('------------------------------------------')
    
    setup()
    pyglet.app.run()

    pyglet.clock.unschedule(game.update)
    if plotting:
        obs.velocity_distribution(game)
        obs.energy(game)
        obs.diffusion(game)
        obs.pair_correlation(game, fig_combined_pc, ax_combined_pc)
        obs.temperature(game)
        plt.show()

    
if __name__ == '__main__':
    plt.close('all')
    fig_combined_pc, ax_combined_pc = plt.subplots()

    main(temperature=1, density=0.8, particles=500, plotting=True)

    
    """Excersise"""
#    main(temperature=0.5, density=1.2, particles=256, plotting=True)
#    main(temperature=1, density=0.8, particles=256, plotting=True)
#    main(temperature=3, density=0.3, particles=256, plotting=True)

    print("Done!")