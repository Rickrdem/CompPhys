import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import functions as func

class Dynamics():
    def __init__(self, J, T, fig, ax):
        self.rows = self.columns = 50
        self.N = self.rows * self.columns
        self.spinchoice = [1,-1]
        self.J = J
        self.T = T
        self.generate_state(fig, ax)
            
        self.m = 0
        self.chi = 0
        
        self.magnetization = []
        self.susceptibility = []
        
    def generate_state(self, fig, ax):
        self.state = np.random.choice(self.spinchoice, (self.rows,self.columns))
        self.state = np.eye(self.rows) * 2 - 1
        # self.im = plt.imshow(self.state, cmap=cm.binary, animated=True)
        
        # cbar = fig.colorbar(self.im, ticks=[-1, 1])
        # cbar.ax.set_yticklabels(['-1', '1'])
    
    def update(self, i=None):
        for i in range(10):
            self.state = func.perturb(self.state, self.J, self.T)
        # self.im.set_array(-self.state)
        
        self.m = np.average(self.state)
        self.chi = self.N *(np.average(np.square(self.state)) - self.m**2)
        
        self.magnetization.append(self.m)
        self.susceptibility.append(self.chi)
        
        print(np.sum(self.state))
        print(self.T)
        
        return #self.im, #<-- What does this , do ?
