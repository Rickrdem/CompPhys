import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup
import matplotlib.pyplot as plt
# from win32api import GetSystemMetrics

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

def plot_pair_correlation(game, fig, ax):
    dr = 1/game.particles**(1/2)
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
    ax.set_xlabel("r/$\sigma$")
    ax.set_ylabel("g(r/$\sigma$)")
    ax.legend()
    fig.tight_layout()
    fig.show()
  
    
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
        plot_velocity_distribution(game)
        plot_energy(game)
        plot_diffusion(game)
        plot_pair_correlation(game, fig_combined_pc, ax_combined_pc)
        plt.show()

    
if __name__ == '__main__':
    plt.close('all')
    fig_combined_pc, ax_combined_pc = plt.subplots()

#    main(plotting=True)
    main(temperature=0.5, density=1.2, particles=256, plotting=True)
    #main(temperature=1, density=0.8, particles=256, plotting=True)
    #main(temperature=3, density=0.3, particles=256, plotting=True)

    print("Done!")