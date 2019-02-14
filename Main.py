import numpy as np
import matplotlib.pyplot as plt
import Functions as func

import pyglet
from gamestate import Gamestate
from viewport import Viewport

N = 60 # Number of particles
L = 1 # Box size
pos = np.random.uniform(0, L, size=(N, 2))

def main(): 
    distances = func.distance(pos, size=(L,L))


if __name__ == '__main__':
    game = Gamestate(100)
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

