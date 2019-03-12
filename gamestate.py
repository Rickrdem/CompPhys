import numpy as np
import functions as func
#import cProfile

class Gamestate():
    def generate_state(self, state):
        """ Generates a standard lattice of positions and velocities (currently fcc)
        :return:
        """
        self.positions = state.copy()
        self.positions_not_bounded = state.copy()
        self.velocities[:] = np.random.normal(0, 0.1,size=(self.particles,self.dimensions))
        self.velocities[:] = self.velocities - np.average(self.velocities, axis=0)[None,:]

    def __init__(self, state, h=0.001, T=0.5, size=(10,10,10), dtype=np.float32):
        self.h = h
        self.m = 1        
        self.T = T
        self.time = 0
        self.dt = 0

        self.size = size
        self.particles = np.shape(state)[0]
        self.dimensions = np.shape(state)[1]

        self.volume = self.size[0]*self.size[1]*self.size[2]
        self.pressure = self.particles * self.T * 119.8 / self.volume
        self.density = self.particles/self.volume
        
        self.dtype = dtype
        self.original_positions = state.copy()

        self.positions = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.velocities = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.forces = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.previous_forces = np.zeros([self.particles,self.dimensions], dtype=dtype)
        self.distances = np.zeros([self.particles, self.particles, self.dimensions], dtype=dtype)
        
        self.potential_energy = []
        self.kinetic_energy = []
        self.diffusion = []       
        self.pair_correlation = []
        
#        self.maxdist = 0
        
        
        self.generate_state(state)

    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """

        self.velocities_update()  # first half
        Lambda = np.sqrt(((self.particles-1)*3*self.T*119.8)/(np.sum(np.square(self.velocities)))) #IS THIS RIGHT?! NOT 1/119.8??!
        self.velocities = Lambda * self.velocities
        
        self.positions_update()
        self.distances_update()
        self.forces_update()
        self.velocities_update()  # seconds half
        

        self.positions[:, :] %= np.asarray(self.size)[np.newaxis, :]
        self.kinetic_energy.append(np.sum(1/2*self.m*np.square(self.velocities))) 

#        self.potential_energy.append(1/2*np.sum(func.U_reduced(func.abs(self.distances))))
        self.potential_energy.append(func.sum_potential_jit(self.distances))

        self.diffusion.append(np.average(np.square((self.positions_not_bounded - self.original_positions))))

        self.pair_correlation.append(func.pair_correlation(self.distances, self.particles, self.volume))


        """Find maximum possible distance between two atoms"""
#        y = np.max(np.sqrt(np.sum(np.square(self.distances), axis=2)))
#        if y > self.maxdist:
#            self.x=y
#        print('max=', self.maxdist) 
        """Increase temparature by predifined time increment"""       
#        self.dt+=2.15*self.h
#        if self.dt > 0.1:
#            self.dt = 0
#            self.T += 0.05
        
        self.time += 2.15*self.h
        
        
    def positions_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.positions[:] = self.positions + self.h*self.velocities
        self.positions_not_bounded[:] = self.positions_not_bounded + self.h*self.velocities

    def velocities_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.velocities[:] = self.velocities + self.h/2 *(self.forces)
        
    def forces_update(self):
        """Update the forces acting on all particles in-place using velocity-verlet"""
        self.forces[:] = np.sum(func.force_jit(self.distances), axis=1)
    
    def distances_update(self):
        """Update the NxNxD distance matrix with the nearest image convention"""
        self.distances[:] = func.distance_jit(self.positions, self.size)


        
        

