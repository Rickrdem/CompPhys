import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics

def plot_energy(game):
    fig, ax = plt.subplots()
    time = np.arange(0, 2.15*game.h*len(game.kinetic_energy), 2.15*game.h)
    ax.plot(time, np.array(game.kinetic_energy[:]),c='r', label='$E_{kin}$')
    ax.plot(time, game.potential_energy, c='b', label='$E_{pot}$')
    ax.plot(time, np.asarray(game.kinetic_energy)+game.potential_energy, c='black', label='H')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy ($1/\epsilon$)")
    ax.legend()
    fig.tight_layout()
    fig.show()

def plot_velocity_distribution(game):
    fig, ax = plt.subplots()
    ax.hist(func.abs(game.velocities), bins=int(game.particles**0.5))
    ax.set_xlabel("Velocity")
    ax.set_ylabel("#")
    fig.tight_layout()
    fig.show()

def plot_diffusion(game):
    fig, ax = plt.subplots()
    time = np.arange(0, 2.15*game.h*len(game.diffusion), 2.15*game.h)
    ax.plot(time, game.diffusion, c='b', label='$Diffusion$')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("$<(x(t)-x(0))^2>$")
    ax.legend()
    fig.tight_layout()
    fig.show()

def plot_pair_correlation_maximum(game):
    fig, ax = plt.subplots()
    time = np.arange(0, 2.15*game.h*len(game.diffusion), 2.15*game.h)
    ax.plot(time, game.pair_correlation, c='b', label='Pair-Correlation maximum')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    fig.show()

def pair_correlation(game, fig, ax):
    dr = 0.05
    radius = np.arange(0.4, game.size[0], dr)
    pair_correlation = []
    for r in radius:
        distances = np.sqrt(np.sum(np.square(game.distances), axis=2))
        X = (distances > r) == True
        Y = (distances < r+dr) == True
        Z = X==Y
        n_r = np.sum(Z, axis=1)
        g_r = 2*game.volume/(game.particles*(game.particles-1))*np.average(n_r)/(4*np.pi*r*r*dr)
        pair_correlation.append(g_r)
      
#    fig, ax = plt.subplots()
    ax.plot(radius, pair_correlation/(np.sum(pair_correlation)*dr), c='b', label='Pair Correlation')
    ax.set_xlabel("Radius")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    fig.show()
  
    
def main(temperature, lattice_constant):
    L = 5
    state = func.fcc_lattice(L, a=lattice_constant)

    game = Gamestate(state, T=temperature, size=(L,L,L), dtype=np.float64)

    print('Game created')
    window = Viewport(game, drawevery=1)
    print('Windows created')

    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    
    window.set_size(width * 2//3, height * 2//3)
    
    window.set_caption('Hexargon')

    pyglet.clock.schedule(game.update)
    print('Starting app')
    
    print('------------------------------------------')   
    print("""
        Particles {particles}
        Boxsize {boxsize}
        Temperature {temp:.2f}
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

    plot_velocity_distribution(game)   
    plot_energy(game)
    plot_diffusion(game)
    
    pair_correlation(game, fig_combined_pc, ax_combined_pc)
    plot_pair_correlation_maximum(game)    
    
if __name__ == '__main__':
    plt.close('all')
    fig_combined_pc, ax_combined_pc = plt.subplots()
    for T in [0.5, 1, 3]:
        if T == 0.5:# Solid
            lattice_constant = 1.4
        elif T == 1: # Liquid
            lattice_constant = 1.6
        elif T== 3: # Gas
            lattice_constant = 2

        main(T, lattice_constant)
    
    plt.show()
    print("Done!")