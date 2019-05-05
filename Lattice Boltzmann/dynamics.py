import numpy as np

class Dynamics():
    def __init__(self, xsize=520, ysize=180):
        self.n_neighbours = 9  # simple square lattice
        self.shape = (xsize, ysize, self.n_neighbours)
        self.flowin = np.zeros(self.shape)
        self.flowout = self.flowin.copy()
        self.floweq = self.flowin.copy()

        self.reynolds = 220.
        self.influx_v = 0.4  # the velocity in lattice units of the ingoing flow

        # self.lattice_velocity = [(x,y) for y in [-1,0,1] for x in [-1,0,1]]
        #
        # self.lattice_weight = [1/9 for i in range(9)]
        # self.lattice_weight[::2] = [1/36 for i in self.lattice_weight[::2]]
        # self.lattice_weight[4] = 4/9

        self.masked_off = np.zeros((self.shape[0], self.shape[1]), dtype=bool)
        self.masked_off[30:60,30:60] = True

    def update(self):
        pass