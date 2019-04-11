import matplotlib.pyplot as plt
from matplotlib.widgets import *
from matplotlib import cm
import time
import matplotlib.animation as animation


class Viewer():
    MINTEMP = 1E-10
    MAXTEMP = 100
    TEMP_STEP = 0.1
    def __init__(self, simulation_state, update_every=1):
        # self.fig, self.ax = plt.subplots()
        self.simulation_state = simulation_state
        self.update_every = update_every

        self._init_matplotlib()

    def update(self, *args):
        self.simulation_state.T = float(self.temp_slider.val)
        for i in range(self.update_every):
            self.simulation_state.update()
        self._update_fig()
        return self.ax,

    def _init_matplotlib(self):
        self.fig = plt.figure()
        self.ax = plt.imshow(self.simulation_state.state, animated=True)
        # animation.FuncAnimation(self.fig, self.simulation_state.update, 1, interval=1, blit=True)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        self.axtemp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.temp_slider = Slider(self.axtemp, 'Temperature', self.MINTEMP, self.MAXTEMP, valinit=1, valstep=self.TEMP_STEP)
        # self.ax.imshow(self.simulation_state.state, cmap=cm.binary, animated=True)
        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=5)

    def _update_fig(self):
        self.ax.set_array(self.simulation_state.state)


if __name__ == '__main__':
    viewer = Viewer(None)
    viewer.fig.show()
    plt.show()