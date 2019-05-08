import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

def density(flow):
    mag = np.sum(flow, axis=-1)
    mag -= np.min(mag)
    mag /= np.max(mag)
    mag = np.nan_to_num(mag)
    return mag

def normalised_magnitude(velocity):
    mag = np.sqrt(np.sum(np.square(velocity), axis=-1))
    mag -= np.min(mag)
    mag /= np.max(mag)
    mag = np.nan_to_num(mag)
    return mag

class Viewer():
    def __init__(self, state):
        self.state = state
        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
        self.ax = plt.imshow(normalised_magnitude(self.state.velocity), cmap=cm.viridis, animated=True)
        self.fig.colorbar(self.ax)

        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=1)

    def update(self, *args):
        self.state.update()
        self.ax.set_array(normalised_magnitude(self.state.velocity))
        return self.ax,
