import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import functions as func

class Dynamics():
    def __init__(self, J, T):
        self.rows = self.columns = 256
        self.N = self.rows * self.columns
        self.spinchoice = [1,-1]
        self.J = J
        self.T = T
        self.magnetic_field = 1

        self.neighbours = np.array([[((i-1)%self.rows, j), ((i+1)%self.rows, j),
                                   (i, (j-1)%self.columns), (i,(j+1)%self.columns)]
                            for i in range(self.rows) for j in range(self.columns)
                           ], dtype=int).reshape(self.rows, self.columns, 4, 2)

        self.generate_state()
            
        self.m = 0
        self.chi = 0
        
        self.magnetization = []
        self.susceptibility = []
        self.energy = []
        
    def generate_state(self):
        self.state = np.random.choice(self.spinchoice, (self.rows,self.columns))
    
    def update(self):
        self.state, energy_chunk = func.metropolis(self.state, self.neighbours, self.T, steps=1)#self.columns*20)
        self.energy.extend(energy_chunk)
        
        self.m = np.average(self.state)
        self.chi = self.N *(np.average(np.square(self.state)) - self.m**2)
        
        self.magnetization.append(self.m)
        self.susceptibility.append(self.chi)
