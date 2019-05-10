import numpy as np
from latticeboltzmann import collision, streaming, flow_equilibrium, update, initial_velocity

class Dynamics():


    def __init__(self, xsize=520, ysize=180):
        self.n_neighbours = 9  # simple square lattice
        self.shape = (xsize, ysize, self.n_neighbours)
        self.flowin = np.zeros(self.shape)
        self.flowout = self.flowin.copy()
        self.floweq = self.flowin.copy()

        self.velocity = np.empty((xsize, ysize, 2))

        self.reynolds = 220.
        self.influx_v = 0.04  # the velocity in lattice units of the ingoing flow

        # self.masked_off = np.zeros((self.shape[0], self.shape[1]), dtype=bool)
        #
        # self.masked_off[10:15,30:40] = True
        # self.masked_off[10:15, 44:60] = True

        xc = xsize / 4
        yc = ysize / 2
        r = ysize/9
        self.masked_off= np.fromfunction(lambda x, y: (x - xc) ** 2 + (y - yc) ** 2 < r ** 2, (xsize, ysize))

        self.init_lattice()

    def init_lattice(self):
        self.velocity = initial_velocity(np.arange(self.velocity.shape[0]), np.arange(self.velocity.shape[1]))#np.fromfunction(lambda x,y,d: np.sin(x/4)/10000 - np.cos(y/20)/10000, (self.shape[0], self.shape[1], 2))
        self.flowin = flow_equilibrium(self.velocity, np.ones((self.shape[0], self.shape[1])))

    def update(self):
        self.flowin, self.velocity = update(self.flowin, self.masked_off, steps=10)
        return self.velocity
