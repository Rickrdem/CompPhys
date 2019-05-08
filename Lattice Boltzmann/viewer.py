<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

# def density(flow):
#     # return flow[:,:,0]
#     return np.sum(flow, axis=-1)

def normalised_magnitude(velocity):
    mag = np.sqrt(np.sum(np.square(velocity), axis=-1))
    mag /= np.average(mag)
    return mag

class Viewer():
    def __init__(self, state):
        self.state = state
=======
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

class Viewer():
    """
    Visualizer.

    This class contains the methods required to visualize the Ising model.

    :param simulation_state (): Contains all parameters of the simulation.
    """
    def __init__(self, simulation_state):
        self.simulation_state = simulation_state

>>>>>>> 06c755dd8013f0b0b23af834902844f74996e8ef
        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
<<<<<<< HEAD
        self.ax = plt.imshow(normalised_magnitude(self.state.velocity) / 5, cmap=cm.viridis, animated=True)

        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=1)

    def update(self, *args):
        self.state.update()
        self.ax.set_array(normalised_magnitude(self.state.velocity))
        return self.ax,
=======
        self.ax = plt.imshow(self.simulation_state.state, cmap=cm.binary, animated=True)
        
        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=1)
        
        print('Windows created')
        print('Starting app')

    def update(self, *args):
        """
        Updates the figure.
        """
        self.simulation_state.update()
        self.ax.set_array(self.simulation_state.state)
        return self.ax,

        
>>>>>>> 06c755dd8013f0b0b23af834902844f74996e8ef
