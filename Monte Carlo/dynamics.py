import numpy as np
import matplotlib.pyplot as plt

import functions as func

class Dynamics():
    def __init__(self, fig, ax):
        self.rows = 100
        self.columns = 50
        self.N = self.rows * self.columns
        self.spinchoice = [1,-1]
        self.J = 0.5
        
        self.generate_state(fig, ax)
        
    def generate_state(self, fig, ax):
        self.state = np.random.choice(self.spinchoice, (self.rows,self.columns))
        self.state = np.eye(self.rows) * 2 - 1
        
        self.im = plt.imshow(self.state, animated=True)
        
        cbar = fig.colorbar(self.im, ticks=[-1, 1])
        cbar.ax.set_yticklabels(['-1', '1'])
    
    def update(self, i=None):
        for i in range(10):
            self.state = func.perturb(self.state, self.J, 2)
        self.im.set_array(self.state)
        return self.im,
