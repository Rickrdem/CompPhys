import matplotlib.pyplot as plt
from matplotlib.widgets import *
from matplotlib import cm
import time
import matplotlib.animation as animation


class Viewer():
    MINTEMP = 1E-10
    MAXTEMP = 4
    TEMP_STEP = 1e-4
    MINCOUPLING = -5
    MAXCOUPLING = 5
    COUPLING_STEP = 0.1
    MINFIELD = -5
    MAXFIELD = 5
    FIELD_STEP = 0.1    
    def __init__(self, simulation_state, update_every=1):
        # self.fig, self.ax = plt.subplots()
        self.simulation_state = simulation_state
        self.update_every = update_every

        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
        self.ax = plt.imshow(self.simulation_state.state, cmap=cm.binary, animated=True)

        plt.subplots_adjust(left=0.25, bottom=0.25)

        self.axtemp = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.temp_slider = Slider(self.axtemp, 'Temperature', self.MINTEMP, self.MAXTEMP, 
                                  valinit=self.simulation_state.T, valstep=self.TEMP_STEP)

        self.axcoupling = self.fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.coupling_slider = Slider(self.axcoupling, 'Pair-Coupling', self.MINCOUPLING, self.MAXCOUPLING, 
                                      valinit=self.simulation_state.J, valstep=self.COUPLING_STEP)
        
        self.axfield = self.fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.field_slider = Slider(self.axfield, 'Magnetic Field', self.MINFIELD, self.MAXFIELD, 
                                      valinit=self.simulation_state.magnetic_field, valstep=self.FIELD_STEP)

        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=5)

    def update(self, *args):
        self.simulation_state.T = float(self.temp_slider.val)
        self.simulation_state.J = float(self.coupling_slider.val)
        for i in range(self.update_every):
            self.simulation_state.update()
        self._update_fig()
        return self.ax,

    def _update_fig(self):
        self.ax.set_array(self.simulation_state.state)


if __name__ == '__main__':
    viewer = Viewer(None)
    viewer.fig.show()
    plt.show()