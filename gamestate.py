import numpy as np
from itertools import product
import functions as func
from viewport import cube, icosahedron

class Gamestate():
    def generate_state(self):
        """ Generates a standard lattice of positions and velocities (currently random)
        :return:
        """
#        self.positions = np.random.uniform(0, self.size[0], size=(self.particles, self.dimensions))# + np.array(self.size)/2
        verts, faces = icosahedron()
        # self.positions[:] = np.array([[10,10,10],[11,11,11]])
        self.positions[:] = verts*3 + np.asarray(self.size)/2
        # self.positions[0,0] += 0.0000001
        # self.positions = func.fcc_lattice(self.size[0], a=1)
        self.positions += np.random.uniform(-.1, .1, size=(self.particles, self.dimensions))
#        self.velocities = np.zeros(shape=(self.particles, self.dimensions))
#         self.velocities[:] = np.random.normal(0, 0,size=(self.particles,self.dimensions))
        self.positions[:,:] %= np.asarray(self.size)[np.newaxis,:]
        
        # self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) # Missing a factor of 2 ?!
        # self.distances_update()
        # self.potential_energy.append(np.sum(func.U_reduced(self.distances)))

        
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

        self.velocities_update()
        self.positions_update()
        self.distances_update()
        self.forces_update()
        self.velocities_update()  # Not required for physics but required for energy

        self.positions[:, :] %= np.asarray(self.size)[np.newaxis, :]

        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) # Missing a factor of 2 ?!
        self.potential_energy.append(np.sum(func.U_reduced(func.abs(self.distances))))
        
        
    def positions_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.positions[:] = self.positions + self.h*self.velocities# + 1/2*self.h**2 * self.forces

    def velocities_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.velocities[:] = self.velocities + self.h/2 *(self.forces)#+self.previous_forces)

    
    def forces_update(self):
        """Update the forces acting on all particles in-place using velocity-verlet"""
        self.forces[:] = np.sum(func.force_reduced(self.distances), axis=1)
    
    def distances_update(self):
        """Update the NxNxD distance matrix with the nearest image convention"""
        self.distances = func.distance_completely_vectorized(self.positions, self.size)
        
        

