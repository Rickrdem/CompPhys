import matplotlib.pyplot as plt
from matplotlib.widgets import *
from matplotlib import cm
import matplotlib.patches as mpatches
import time
import matplotlib.animation as animation


class Viewer():
    MINTEMP = 1E-10
    MAXTEMP = 4
    TEMP_STEP = 1e-4
    MINCOUPLING = -5
    MAXCOUPLING = 5
    COUPLING_STEP = 0.01
    MINFIELD = -5
    MAXFIELD = 5
    FIELD_STEP = 0.01    
    def __init__(self, simulation_state, update_every=1):
        # self.fig, self.ax = plt.subplots()
        self.simulation_state = simulation_state
        self.update_every = update_every
        
        
        self.initial_temperature = self.simulation_state.T
        self.initial_paircoupling = self.simulation_state.J
        self.initial_field = self.simulation_state.magnetic_field


        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
        self.ax = plt.imshow(self.simulation_state.state, cmap=cm.binary, animated=True)

        plt.subplots_adjust(left=0.25, bottom=0.25)

        values = self.simulation_state.spinchoice
        colors = [ self.ax.cmap(self.ax.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="Spin {l}".format(l=values[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fancybox=True, shadow=True )
        
        #Siders
        self.axspeed = self.fig.add_axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.speed_slider = Slider(self.axspeed, 'Speed', 1, 5000, 
                                  valinit=self.simulation_state.steps_per_refresh, valstep=1, valfmt="%1.0i")

        self.axtemp = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.temp_slider = Slider(self.axtemp, 'Temperature', self.MINTEMP, self.MAXTEMP, 
                                  valinit=self.simulation_state.T, valstep=self.TEMP_STEP)

        self.axcoupling = self.fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.coupling_slider = Slider(self.axcoupling, 'Pair-Coupling', self.MINCOUPLING, self.MAXCOUPLING, 
                                      valinit=self.simulation_state.J, valstep=self.COUPLING_STEP)
        
        self.axfield = self.fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.field_slider = Slider(self.axfield, 'Magnetic Field', self.MINFIELD, self.MAXFIELD, 
                                      valinit=self.simulation_state.magnetic_field, valstep=self.FIELD_STEP)

        
        self.resetax = self.fig.add_axes([0.85, 0.25, 0.1, 0.04])
        self.reset_button = Button(self.resetax, 'Reset', color='red', hovercolor='0.975')

        self.algorithm_ax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor='white')
        self.algoritm_menu = RadioButtons(self.algorithm_ax, ('Metropolis', 'Wolff'), active=0)


        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=5)

    def update(self, *args):
        self.simulation_state.steps_per_refresh = int(self.speed_slider.val)
        self.simulation_state.T = float(self.temp_slider.val)
        self.simulation_state.J = float(self.coupling_slider.val)
        self.simulation_state.magnetic_field = float(self.field_slider.val)
        self.reset_button.on_clicked(self.reset)
        self.algoritm_menu.on_clicked(self.alogorithm_choice)
        
        for i in range(self.update_every):
            self.simulation_state.update()
        self._update_fig()
        return self.ax,

    def _update_fig(self):
        self.ax.set_array(self.simulation_state.state)

    def reset(self, event):
        self.simulation_state.T = self.initial_temperature
        self.simulation_state.J = self.initial_paircoupling
        self.simulation_state.magnetic_field = self.initial_field
        self.temp_slider.reset()
        self.coupling_slider.reset()
        self.field_slider.reset()

    def alogorithm_choice(self, label):
#        print(label)
        if label == 'Wolff':
            print("Wolff activated")
#            self.simulation_state.wolff_algorithm = True
#            self.simulation_state.metropolis_algorithm = False
        elif label == 'Metropolis':
            print("Metropopis activated")
#            self.simulation_state.wolff_algorithm = False
#            self.simulation_state.metropolis_algorithm = True

#        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    viewer = Viewer(None)
    viewer.fig.show()
    plt.show()