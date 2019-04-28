"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np

import monte_carlo as mc
import observables as obs

class Dynamics():
    """
    Dynamics integrator.

    This class contains the methods required to simulate the Ising model.
    Typical usage is to initialise with a state and propagate that state with the update method.

    :param j_coupling (float): The initial spin-spin coupling.
    :param temp (float): The initial temperature.
    """

    def __init__(self, j_coupling, temp, magnetic_field=0, algorithm='Metropolis'):
        self.rows = self.columns = 100
        self.num_spins = self.rows * self.columns
        self.spinchoice = [1,-1]
        self.j_coupling = j_coupling
        self.temp = temp
        self.magnetic_field = magnetic_field
        
        self.steps_per_refresh = 100

        self.algorithm_selected = algorithm
        self.algorithm_options = {'Metropolis':mc.metropolis, 
                                  'Wolff':mc.wolff, 
                                  'Heat Bath':mc.heat_bath,
                                  'Checkerboard':mc.checkerboard}

        self.neighbours = np.array([[((i-1)%self.rows, j), ((i+1)%self.rows, j),
                                   (i, (j-1)%self.columns), (i,(j+1)%self.columns)]
                            for i in range(self.rows) for j in range(self.columns)
                           ], dtype=int).reshape(self.rows, self.columns, 4, 2)

        self.neighbours = np.array([[(i//self.columns) * self.rows + (i+1) % self.rows,
                                     (i+self.columns) % self.num_spins,
                                     (i//self.columns) * self.rows + (i-1) % self.rows,
                                     (i-self.columns) % self.num_spins]
                                    for i in range(self.num_spins)])

        self.generate_state()

        self.magnetization = []
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
        self.state = self.algorithm_options[self.algorithm_selected](self.state.flatten(), self.neighbours, self.temp,
                                                                     self.j_coupling, self.magnetic_field, steps=self.steps_per_refresh)
        
        self.state = self.state.reshape(self.rows,-1)
        
        self.magnetization.append(np.sum(self.state) / self.num_spins)
        self.energy.append(obs.energy(self.state, self.j_coupling, self.magnetic_field))