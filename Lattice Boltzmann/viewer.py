import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

def density(flow):
    mag = np.sum(flow, axis=-1)
    return mag

def normalised_magnitude(velocity):
    mag = np.sqrt(np.sum(np.square(velocity), axis=-1))
    return mag

class Viewer():
    """
    Visualizer.

    This class contains the methods required to visualize the Lattice Boltzman method.

    :param simulation_state (): Contains all parameters of the simulation.
    """
    MAXFRAMES = 1800  # 30 sec@60fps

    def __init__(self, state):
        self.state = state
        self.save_video = False
        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure(figsize=(10,3))
        ax = plt.gca()
        self.ax = ax.imshow(normalised_magnitude(self.state.velocity).T, cmap=cm.Reds, animated=True, vmin=0., vmax=0.08)
        self.fig.colorbar(self.ax, ax=ax)

        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=1, save_count=1800)
#        plt.show()

        if self.save_video:  # Requires ffmpeg package
            now = datetime.datetime.now()
            writer_init = animation.writers['ffmpeg']
            writer = writer_init(fps=60, metadata=dict(artist='Me'), bitrate=1800)
            self.animation_handler.save('{y}{m}{d}_{h}{min}{s}.mp4'.format(y=now.year,
                                                                           m=now.month,
                                                                           d=now.day,
                                                                           h=now.hour,
                                                                           min=now.minute,
                                                                           s=now.second), writer=writer)


    def update(self, *args):
        self.state.update()
        self.ax.set_array(normalised_magnitude(self.state.velocity).T)
        return self.ax,
