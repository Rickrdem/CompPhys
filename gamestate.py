import numpy as np
from itertools import product
import functions as func

class Gamestate():
    def generate_state(self):
        """ Generates a standard lattice of positions and velocities (currently random)
        :return:
        """
#        self.positions = np.random.uniform(0, self.size[0], size=(self.particles, self.dimensions))# + np.array(self.size)/2
        self.positions = func.fcc_lattice(self.size[0], a=1)
        self.positions[:] = self.positions + np.random.uniform(-.1, .1, size=(self.particles, self.dimensions))
#        self.velocities = np.zeros(shape=(self.particles, self.dimensions))
        self.velocities[:] = np.random.normal(0, 100,size=(self.particles,self.dimensions))
        self.positions[:,:] %= np.asarray(self.size)[np.newaxis,:]
        
        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) # Missing a factor of 2 ?!
        self.distances_update()
        self.potential_energy.append(np.sum(func.U_reduced(self.distances)))

        
        print(self.kinetic_energy)
        
#        self.directions = np.angle(self.velocities[:,0]+1j*self.velocities[:,1])

    def __init__(self, particles=100, h=0.001, size=(10,10,10), dtype=np.float32):
        self.particles = particles
        self.size = size
        self.dimensions = len(size)
        self.h = h
        self.m = 1        
        
        self.dtype = dtype
        
        self.positions = np.zeros([particles,self.dimensions], dtype=dtype)
        self.velocities = np.zeros([particles,self.dimensions], dtype=dtype)
        self.forces = np.zeros([particles,self.dimensions], dtype=dtype)
        self.previous_forces = np.zeros([particles,self.dimensions], dtype=dtype)
        self.distances = np.zeros([particles, particles, self.dimensions], dtype=dtype)
        
        self.potential_energy = []
        self.kinetic_energy = []
        
        
        self.generate_state()

    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """
        
        
        self.positions_update()        
        self.distances_update()
        self.forces_update()
        self.velocities_update()

        
        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) # Missing a factor of 2 ?!
        self.potential_energy.append(np.sum(func.U_reduced(self.distances)))

        
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
#        self.forces[:] = np.sum(func.absolute_force_reduced(func.distance_completely_vectorized(self.positions, self.size)), axis=0)
    
    def distances_update(self):
        self.distances = func.distance_matrix(self.positions, self.size)
        
        

