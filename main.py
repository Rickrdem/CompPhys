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
   
def pair_correlation():
    radius = np.arange(0.34, game.size[0], 0.05)
    dr = 0.3
    pair_correlation = []
    for r in radius:
        distances = np.sqrt(np.sum(np.square(game.distances), axis=2))
        X = (distances > r) == True
        Y = (distances < r+dr) == True
        Z = X==Y
    
        n_r = np.average(np.sum(Z, axis=1))
        
        g_r = 2*game.volume/(game.particles*(game.particles-1))*n_r/(4*np.pi*r*r*dr)
    
        pair_correlation.append(g_r)
    
    fig, ax = plt.subplots()
    ax.plot(radius, pair_correlation/(np.sum(pair_correlation)*dr), c='b', label='Pair Correlation')
    ax.set_xlabel("Radius")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    fig.show()
 
    
    
if __name__ == '__main__':
    L = 5
    lattice_constant = 1.1
    state = func.fcc_lattice(L, a=lattice_constant)

    game = Gamestate(state, T=0.2, size=(L,L,L), dtype=np.float64)

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

    plot_velocity_distribution()   
    plot_energy()
    plot_diffusion()
    pair_correlation()

    print("Done!")