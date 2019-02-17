import numpy as np
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
#        self.positions = np.random.uniform(0, L, size=(N, dim))
        self.velocities = np.zeros(shape=(N, dim))
        
        self.directions = np.angle(self.velocities[:,0]+1j*self.velocities[:,1])

    def __init__(self, particles=2000, size=(300, 300), drawevery=1):
        self.particles = particles
        self.size = size
        # self.dimensions = dimensions
        self.dimensions = len(size)
        self.drawevery = drawevery

        self.generate_state()

    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """
        self.positions = self.positions + self.velocities  #Move particles around

        dim = self.dimensions
        L = self.size[0]
        h = 0.0001 # Timestep (s) 
        m = 0.001 # Mass (set to unity)
        
        direction_vector = func.distance_completely_vectorized(self.positions, dim*[L])
        
        distance = np.sqrt(np.sum(np.square(direction_vector), axis=2))
        
        
        force = func.absolute_force(distance, sigma = 0.1, epsilon = 1000)
        force[force==np.inf] = 0
        
        
        force_vector = direction_vector * force[:,:,np.newaxis] # Force of each point on each other point, in each direction
        total_force = np.sum(force_vector, axis = 1) # Total force on each particle (magnitude and direction)
    
        
        x_n_plus_1 = self.positions + self.velocities * h 
        v_n_plus_1 = self.velocities + 1 / m * total_force * h
        
        
        self.positions = x_n_plus_1
        self.velocities = v_n_plus_1

        self.positions[:, 0] %= self.size[0]  #Modulo the size of the space
        self.positions[:, 1] %= self.size[1]

