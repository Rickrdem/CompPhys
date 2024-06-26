import numpy as np
from latticeboltzmann import collision, streaming, flow_equilibrium, update, initial_velocity

class Dynamics():


    def __init__(self, xsize=520, ysize=180, show_every=10):
        self.n_neighbours = 9  # simple square lattice
        self.shape = (xsize, ysize, self.n_neighbours)
        self.flowin = np.zeros(self.shape)
        self.flowout = self.flowin.copy()
        self.floweq = self.flowin.copy()

        self.velocity = np.empty((xsize, ysize, 2))

        self.reynolds = 220.
        self.influx_v = 0.04  # the velocity in lattice units of the ingoing flow

        self.steps_per_update = show_every
        self.elapsed_time = 0


        #  Single obstable
        xc = xsize / 4
        yc = ysize / 2
        r = ysize/9
        self.masked_off = np.fromfunction(lambda x, y: (x - xc) ** 2 + (y - yc) ** 2 < r ** 2, (xsize, ysize))


        #  Multiple obstacles
        # xc = xsize / 4
        # yc = ysize / 10
        # r = ysize/9
        # self.masked_off1 = np.fromfunction(lambda x, y: (x - xc) ** 2 + (y - 3*yc) ** 2 < r ** 2, (xsize, ysize))
        # self.masked_off2 = np.fromfunction(lambda x, y: (x - xc) ** 2 + (y - 7*yc) ** 2 < r ** 2, (xsize, ysize))
        # self.masked_off = np.any((self.masked_off1, self.masked_off2), axis=0)


        self.masked_off[:, 0] = 1
        self.masked_off[:, -1] = 1


        self.init_lattice()

    def init_lattice(self):
        self.velocity = initial_velocity(np.arange(self.velocity.shape[0]), np.arange(self.velocity.shape[1]))#np.fromfunction(lambda x,y,d: np.sin(x/4)/10000 - np.cos(y/20)/10000, (self.shape[0], self.shape[1], 2))
        self.flowin = flow_equilibrium(self.velocity, np.ones((self.shape[0], self.shape[1])))

    def update(self):
        self.flowin, self.velocity = update(self.flowin, self.masked_off, steps=self.steps_per_update)
        self.elapsed_time += self.steps_per_update
        print(self.elapsed_time)
        return self.velocity
