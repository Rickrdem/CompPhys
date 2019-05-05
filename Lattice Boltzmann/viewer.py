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

        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
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

        