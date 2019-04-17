"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import monte_carlo as mc

class Dynamics():
    """
    Dynamics integrator.

    This class contains the methods required to simulate the Ising model.
    Typical usage is to initialise with a state and propagate that state with the update method.

    :param J (float): The initial pair-pair coupling.
    :param T (float): The initial temperature.
    """

    def __init__(self, J, T):
        self.rows = self.columns = 100
        self.N = self.rows * self.columns
        self.spinchoice = [1,-1]
        self.J = J
        self.T = T
        self.magnetic_field = 0
        
        self.steps_per_refresh = 100

        self.metropolis_algorithm = True
        self.wolff_algorithm = False

        self.neighbours = np.array([[((i-1)%self.rows, j), ((i+1)%self.rows, j),
                                   (i, (j-1)%self.columns), (i,(j+1)%self.columns)]
                            for i in range(self.rows) for j in range(self.columns)
                           ], dtype=int).reshape(self.rows, self.columns, 4, 2)

        self.neighbours = np.array([[(i//self.columns)*self.rows+(i+1)%self.rows,
                                    (i+self.columns)%self.N,
                                    (i//self.columns)*self.rows+(i-1)%self.rows,
                                    (i-self.columns)%self.N]
                                    for i in range(self.N)])

        self.generate_state()
            
        self.m = 0
        self.chi = 0
        
        self.magnetization = []
        self.susceptibility = []
        self.energy = []
        
    def generate_state(self):
        """ 
        Generates the initial state.
        """
        self.state = np.random.choice(self.spinchoice, (self.rows,self.columns))
    
    def update_and_save(self):
        """ 
        Updates the dynamic state and saves macroscopic parameters.
        """
        if self.metropolis_algorithm:
            self.state = mc.metropolis(self.state.flatten(), self.neighbours, self.T, self. J, self.magnetic_field, steps=self.steps_per_refresh)
        elif self.wolff_algorithm:
            self.state = mc.wolff(self.state.flatten(), self.neighbours, self.T, self.J, self.magnetic_field, steps=self.steps_per_refresh)
        
        self.state = self.state.reshape(self.rows,-1)
        # self.energy.extend(energy_chunk)
        
        self.m = np.average(self.state)
        self.chi = self.N *(np.average(np.square(self.state)) - self.m**2)
        
        self.magnetization.append(self.m)
        self.susceptibility.append(self.chi)