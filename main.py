import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup
import matplotlib.pyplot as plt

plt.close('all')

def plot_energy():
    fig, ax = plt.subplots()
    time = np.arange(0, game.h*len(game.kinetic_energy), game.h)
    ax.plot(time, 2*np.array(game.kinetic_energy[:]),c='r', label='$E_{kin}$')
    ax.plot(time, game.potential_energy, c='b', label='$E_{pot}$')
    ax.plot(time, 2*np.asarray(game.kinetic_energy)+game.potential_energy, c='black', label='H')
    ax.set_xlabel("Time ($(m\sigma^2/\epsilon)^{1/2}$)")
    ax.set_ylabel("Energy ($1/\epsilon$)")
    ax.legend()
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    L = 7
    game = Gamestate(particles=12, h=0.0001, size=(L,L,L))

    # game.update(1)

    print('Game created')
    window = Viewport(game, drawevery=200)
    print('Windows created')
    window.set_size(1280 * 2//3, 720 * 2//3)

    pyglet.clock.schedule(game.update)
    print('Starting app')
    setup()
    pyglet.app.run()

    pyglet.clock.unschedule(game.update)
    
    plot_energy()
    plt.show()

    
#    for i in range(500):
#        sigma = 3.508*10**(-10)  # meter
#        epsilon_over_kB = 119.8 # Kelvin
#        
#        N = 100 # Number of particles
#        L = 10 # Box size; in units of sigma
#        h = 0.001 # Timestep (s) 
#        m = 1 # Mass (set to unity)
#        dim = 2
#        pos = np.random.uniform(0, L, size=(N, dim))
#    
#        
#        direction_vector = func.distance_completely_vectorized(pos, dim*[L])
#        
#        distance = np.sqrt(np.sum(np.square(direction_vector), axis=2))
#        
#        
#        force = func.absolute_force_reduced(distance)
#        
#    
#        force_vector = direction_vector * force[:,:,np.newaxis] 
#        total_force = np.sum(force_vector, axis = 1)
#    
#        x_n = pos
#        v_n = 0
#        
#        x_n_plus_1 = x_n + v_n * h 
#        v_n_plus_1 = v_n + 1 / m * force_vector * h
#
#    
#    x = np.arange(0.6, 10, 0.01)
#    U_x = func.U_reduced(x)
#    plt.plot(x,U_x)
#    F_x = func.absolute_force_reduced(x)
#    plt.plot(x, F_x)    
#    plt.xlim((0,5))
#    plt.ylim(-15,10)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Done!")    

