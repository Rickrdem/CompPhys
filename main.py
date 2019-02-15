import numpy as np
import matplotlib.pyplot as plt
import functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport

N = 3 # Number of particles
L = 1 # Box size
h = 40 # Timestep (s) 
m = 1 # Mass (set to unity)
pos = np.random.uniform(0, L, size=(N, 2))

def main(): 
    distances = func.distance(pos, size=(L,L))


if __name__ == '__main__':
    game = Gamestate(1000)
    window = Viewport(game)

    window.set_size(1280 * 2 // 3, 720 * 2 // 3)

    pyglet.clock.schedule(game.update)
    pyglet.app.run()

    pyglet.clock.unschedule(game.update)
    print('done')

    # result = func.distance(pos, size=(L,L))
    #
    # x = np.argmin(result[0,:][np.nonzero(result[0,:])])+1
    # print(x)
    # plt.scatter(pos[:,0], pos[:,1], color='blue')
    # plt.scatter(pos[0,0], pos[0,1], color='red')
    # plt.scatter(pos[x,0], pos[x,1], color='green')
    # plt.xlim((0,L))
    # plt.ylim((0,L))
    #
    # print(result)

    
    
    L = 1 #Box size
    dim = 2
    h = 0.1 # Timestep (s) 
    m = 1 # Mass (set to unity)
    pos = np.random.uniform(0, L, size=(N, dim))
    
    direction_vector = func.distance_completely_vectorized(pos, dim*[L])
    
    distance = np.sqrt(np.sum(np.square(direction_vector), axis=2))
    
    
    force = func.absolute_force(distance, sigma = 0.1, epsilon = 0.1)
    force[force==np.inf] = 0
    
    
    force_vector = direction_vector * force[:,:,np.newaxis] 
    total_force = np.sum(force_vector, axis = 1)

    x_n = pos
    v_n = 0
    
    x_n_plus_1 = x_n + v_n * h 
    v_n_plus_1 = v_n + 1 / m * force_vector * h
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
