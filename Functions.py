import numpy as np
#import pyglet
import matplotlib.pyplot as plt

N = 60 # Number of particles
L = 1 # Box size
pos = np.random.uniform(0, L, size=(N, 2))


#@numba.njit(fastmath=True, parallel=True, debug=False)
def distance(positions, size=(L,L)):
    """Calculate the distance in 2D between each 2 points according to the 
    minimum image convention"""
    
    shape = len(positions)
    result = np.empty((shape,shape))
    for i in range(shape):
        result[i,i] = 0.0
        for j in range(i+1,shape):
            dx = Dist(positions[i,0], positions[j,0], size[0])
            dy = Dist(positions[i,1], positions[j,1], size[1])
            #dz = Dist(position[i,2], position[j,2], size[2]) Something 3D with extra for loop may take too long

            result[i,j] = result[j,i] = (dx**2+dy**2)**(0.5)
        
    return result

def Dist(x1, x2, L):
    """Calculate the distance in 1D between x1 and x2 according to the minimum 
    image convention in a box of length L"""
    dx = min(np.abs(x1 - (x2 + L)),
             np.abs(x1 - x2), 
             np.abs(x1 - (x2 - L)))

    return dx

result = distance(pos)

x = np.argmin(result[0,:][np.nonzero(result[0,:])])+1
print(x)
plt.scatter(pos[:,0], pos[:,1], color='blue')
plt.scatter(pos[0,0], pos[0,1], color='red')
plt.scatter(pos[x,0], pos[x,1], color='green')
plt.xlim((0,L))
plt.ylim((0,L))

print(result)
