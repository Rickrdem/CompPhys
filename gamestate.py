import numpy as np
from itertools import product
import functions as func

class Gamestate():
    def generate_state(self):
        """ Generates a standard lattice of positions and velocities (currently random)
        :return:
        """
#        self.positions = np.random.randn(self.particles, self.dimensions) + np.array(self.size)/2
#        self.velocities = np.random.randn(self.particles, self.dimensions)
        L = self.size[0] #Box size
        dim = self.dimensions
        N = self.particles

        self.positions = np.random.uniform(0, L, size=(N, dim))# + np.array(self.size)/2
        self.positions = func.sc_lattice(L)
        self.velocities = np.zeros(shape=(N, dim))
        self.positions[:,:] %= np.asarray(self.size)[np.newaxis,:]
        
#        self.directions = np.angle(self.velocities[:,0]+1j*self.velocities[:,1])

    def __init__(self, particles=100, h=0.001, size=(300, 300), drawevery=1, dtype=np.float32):
        self.particles = particles
        self.size = size
        self.dimensions = len(size)
        self.h = h
        self.m = 1
        
        self.drawevery = drawevery
        self.dtype = dtype
        
        self.positions = np.zeros([particles,self.dimensions], dtype=dtype)
        self.velocities = np.zeros([particles,self.dimensions], dtype=dtype)
        self.forces = np.zeros([particles,self.dimensions], dtype=dtype)
        self.previous_forces = np.zeros([particles,self.dimensions], dtype=dtype)
        self.distances = np.zeros([particles, particles, self.dimensions], dtype=dtype)
        
        self.MS_velocity = []
        self.potential_energy = []
        self.kinetic_energy = []
        
        
        self.generate_state()

    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """
        
        
        self.distances_update()
        self.forces_update()
        self.positions_update()
        self.velocities_update()
        
#        direction_vector = func.distance_completely_vectorized(self.positions, dim*[L])
        
#        distance = np.sqrt(np.sum(np.square(direction_vector), axis=2))
        

#        force = func.absolute_force(distance, sigma = 1, epsilon = 1000)
#        force = func.absolute_force_reduced(distance)

        
#        force_vector = direction_vector * force[:,:,np.newaxis] # Force of each point on each other point, in each direction
#        total_force = np.sum(force_vector, axis = 1) # Total force on each particle (magnitude and direction)
#        print(np.max(total_force))
        
#        x_n_plus_1 = self.positions + self.velocities * h 
#        v_n_plus_1 = self.velocities + 1 / m * total_force * h
        
        
#        self.positions = x_n_plus_1
#        self.velocities = v_n_plus_1

        
        
        


        mean_square_velocity = np.sqrt(np.sum(np.square(self.velocities)))
        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities)))
        self.MS_velocity.append(mean_square_velocity)
#        self.kinetic_energy.append(1/2 * self.particles * self.m * mean_square_velocity ** 2)  # T = 1/2 N*m*<v>^2
        self.potential_energy.append(np.sum(func.U_reduced(self.distances)))
        

        
        
    def positions_update(self):
        """In place update the positions of all particles using verlet"""
        self.positions[:] = self.positions + self.h*self.velocities + 1/2*self.h**2 * self.forces
        self.positions[:,:] %= np.asarray(self.size)[np.newaxis,:]
    
    def velocities_update(self):
        self.velocities[:] = self.velocities + self.h/2 *(self.forces+self.previous_forces)
    
    def forces_update(self):
        self.previous_forces = self.forces
        self.forces[:] = np.sum(
                                func.distance_completely_vectorized(self.positions, self.size) * func.absolute_force_reduced(self.distances)[:,:,np.newaxis],
                                axis=1
                                )
    
    def distances_update(self):
        self.distances = func.distance_matrix(self.positions, self.size)
        
        

