import numpy as np
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport, setup

sigma = 3.508*10**(-10)  # meter
epsilon_over_kB = 119.8 # Kelvin

N = 2 # Number of particles
L = 50 # Box size; in units of sigma
h = 0.001 # Timestep (s) 
m = 1 # Mass (set to unity)
dim = 2
pos = np.random.uniform(0, L, size=(N, dim))


if __name__ == '__main__':
    L = 10
    game = Gamestate(particles=100, size=(L,L,L))
    print('Game created')
    window = Viewport(game)
    print('Windows created')
    window.set_size(1280 * 2 // 3, 720 * 2 // 3)

    pyglet.clock.schedule(game.update)
    print('Starting app')
    setup()
    pyglet.app.run()

    pyglet.clock.unschedule(game.update)
    print('done')

    
    direction_vector = func.distance_completely_vectorized(pos, dim*[L])
    
    distance = np.sqrt(np.sum(np.square(direction_vector), axis=2))
    
    
    force = func.absolute_force(distance, sigma = 1, epsilon = 1000)
    
    
    force_vector = direction_vector * force[:,:,np.newaxis] 
    total_force = np.sum(force_vector, axis = 1)

    x_n = pos
    v_n = 0
    
    x_n_plus_1 = x_n + v_n * h 
    v_n_plus_1 = v_n + 1 / m * force_vector * h
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
