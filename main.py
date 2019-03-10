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

def plot_velocity_distribution():
    fig, ax = plt.subplots()
    ax.hist(func.abs(game.velocities), bins=int(game.particles**0.5))
    ax.set_xlabel("Velocity")
    ax.set_ylabel("#")
    fig.tight_layout()
    fig.show()

def plot_diffusion():
    fig, ax = plt.subplots()
    time = 2.15*np.arange(0, game.h*len(game.diffusion), game.h)
    ax.plot(time, game.diffusion, c='b', label='$Diffusion$')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("$<(x(t)-x(0))^2>$")
    ax.legend()
    fig.tight_layout()
    fig.show()
   
if __name__ == '__main__':
    L = 5
    lattice_constant = 1.3
    state = func.fcc_lattice(L, a=lattice_constant)

    game = Gamestate(state, T=0.24, size=(L,L,L), dtype=np.float64)

    print('Game created')
    window = Viewport(game, drawevery=1)
    print('Windows created')
    window.set_size(1280 * 2//1, 720 * 2//1)
    window.set_caption('Molecular Dynamics Simulation of Argon')
    pyglet.clock.schedule(game.update)
    print('Starting app')
    
    print('------------------------------------------')   
    print("""
        Particles {particles}
        Boxsize {boxsize}
        Temperature {temp:.2f} K
        Density {dens:.2f}
        Pressure {pressure:.2f}
        """.format(particles=game.particles,
                   boxsize=game.size,
                   temp=game.T,
                   dens=game.density,
                   pressure=game.pressure))
    print('------------------------------------------')
    
    setup()
    pyglet.app.run()

    pyglet.clock.unschedule(game.update)
    
    plot_energy()
    plot_diffusion()
    plot_velocity_distribution()

    print("Done!")