import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup
import matplotlib.pyplot as plt

plt.close('all')

def plot_energy():
    fig, ax = plt.subplots()
    time = 2.15*np.arange(0, game.h*len(game.kinetic_energy), game.h)
    ax.plot(time, np.array(game.kinetic_energy[:]),c='r', label='$E_{kin}$')
    ax.plot(time, game.potential_energy, c='b', label='$E_{pot}$')
    ax.plot(time, np.asarray(game.kinetic_energy)+game.potential_energy, c='black', label='H')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy ($1/\epsilon$)")
    ax.legend()
    fig.tight_layout()
    fig.show()
#    plt.hist(func.abs(game.velocities), bins=int(game.particles**0.5))
    plt.show()

if __name__ == '__main__':
    L = 5
    lattice_constant = 1.4
    state = func.fcc_lattice(L, a=lattice_constant)

    game = Gamestate(state, T=0.5, lattice_constant=1.4, size=(L,L,L), dtype=np.float64)

    # game.update(1)

    print('Game created')
    window = Viewport(game, drawevery=1)
    print('Windows created')
    window.set_size(1280 * 2//2, 720 * 2//2)

    pyglet.clock.schedule(game.update)
    print('Starting app')
    print('------------------------------------------')   
    print('Number of particles:', game.particles)
    print('Pressure:', game.pressure)
    print('Density:', game.density)
    setup()
    pyglet.app.run()
    print('Average velocity:', np.sum(func.abs(game.velocities))/game.particles)

    pyglet.clock.unschedule(game.update)
    
    plot_energy()
    plt.show()
    x=game.velocities
    print('------------------------------------------')
    print("Done!")    

