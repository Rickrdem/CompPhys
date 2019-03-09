import numpy as np
from itertools import product
import functions as func
from viewport import cube, icosahedron
import cProfile

class Gamestate():
    def generate_state(self, state):
        """ Generates a standard lattice of positions and velocities (currently random)
        :return:
        """
        self.positions = state
        self.velocities[:] = np.random.normal(0, 0.1,size=(self.particles,self.dimensions))
        self.velocities[:] = self.velocities - np.average(self.velocities, axis=0)[None,:]

    def __init__(self, state, h=0.001, T=0.5, lattice_constant=1, size=(10,10,10), dtype=np.float32):
        self.size = size
        self.particles = np.shape(state)[0]
        self.dimensions = np.shape(state)[1]
        self.h = h
        self.m = 1        
        self.T = T
        self.time = 0
        self.pressure = 1
        self.density = self.particles/(self.size[0]*self.size[0]*self.size[0])
        
        
        self.dtype = dtype
        
        self.positions = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.velocities = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.forces = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.previous_forces = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.distances = np.zeros([self.particles, self.particles, self.dimensions], dtype=dtype)
        
        self.potential_energy = []
        self.kinetic_energy = []
        
        
        self.generate_state(state)


    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """
        Lambda = np.sqrt(((self.particles-1)*self.T*118.9)/(np.sum(np.square(self.velocities))))

        self.velocities_update()  # first half
        self.velocities = Lambda * self.velocities

        self.positions_update()
        self.distances_update()
        self.forces_update()
        self.velocities_update()  # seconds half
        
        Lambda = np.sqrt(((self.particles-1)*self.T*118.9)/(np.sum(np.square(self.velocities))))
        self.velocities = Lambda * self.velocities


        self.positions[:, :] %= np.asarray(self.size)[np.newaxis, :]

        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) 

        self.potential_energy.append(1/2*np.sum(func.U_reduced(func.abs(self.distances))))
#        self.potential_energy.append(func.sum_potential_jit(self.distances))

        
        
        self.time += self.h
        
        
    def positions_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.positions[:] = self.positions + self.h*self.velocities# + 1/2*self.h**2 * self.forces

    def velocities_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.velocities[:] = self.velocities + self.h/2 *(self.forces)#+self.previous_forces)

    
    def forces_update(self):
        """Update the forces acting on all particles in-place using velocity-verlet"""
        self.forces[:] = np.sum(func.force_jit(self.distances), axis=1)
        # print('total force',np.sum(self.forces))
    
    def distances_update(self):
        """Update the NxNxD distance matrix with the nearest image convention"""
        self.distances[:] = func.distance_jit(self.positions, self.size)
        # self.distances += np.transpose(self.distances, axes=(1,0,2))  # explicitly symmetrize


        
        

