import numpy as np


class Gamestate():
    def generate_state(self):
        """ Generates a standard lattice of positions and velocities (currently random)
        :return:
        """
        self.positions = np.random.randn(self.particles, self.dimensions) + np.array(self.size)/2
        self.velocities = np.random.randn(self.particles, self.dimensions)
        self.directions = np.angle(self.velocities[:,0]+1j*self.velocities[:,1])

    def __init__(self, particles=2000, size=(500, 500), drawevery=1):
        self.particles = particles
        self.size = size
        # self.dimensions = dimensions
        self.dimensions = 2
        self.drawevery = drawevery

        self.generate_state()

    def update(self, a):
        """ Updates the gamestate

        Moves all the particles around
        :return:
        """
        self.positions = self.positions + self.velocities  #Move particles around

        self.positions[:, 0] %= self.size[0]  #Modulo the size of the space
        self.positions[:, 1] %= self.size[1]
