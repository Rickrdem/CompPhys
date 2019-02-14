import numpy as np
import matplotlib.pyplot as plt

import Functions as func

N = 60 # Number of particles
L = 1 # Box size
pos = np.random.uniform(0, L, size=(N, 2))

def main(): 
    distances = func.distance(pos, size=(L,L))


if __name__ == '__main__':
    result = func.distance(pos, size=(L,L))

    x = np.argmin(result[0,:][np.nonzero(result[0,:])])+1
    print(x)
    plt.scatter(pos[:,0], pos[:,1], color='blue')
    plt.scatter(pos[0,0], pos[0,1], color='red')
    plt.scatter(pos[x,0], pos[x,1], color='green')
    plt.xlim((0,L))
    plt.ylim((0,L))
    
    print(result)

