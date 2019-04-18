"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
import matplotlib.patches as mpatches
#import time
import matplotlib.animation as animation

class Viewer():
    """
    Visualizer.

    This class contains the methods required to visualize the Ising model.

    :param simulation_state (): Contains all parameters of the simulation.
    :param update_every (int): Number of loops before the figure is updated.
    """

    MINTEMP = 1E-10
    MAXTEMP = 4
    TEMP_STEP = 1e-4
    MINCOUPLING = -5
    MAXCOUPLING = 200
    COUPLING_STEP = 0.01
    MINFIELD = -5
    MAXFIELD = 5
    FIELD_STEP = 0.01    
    def __init__(self, simulation_state, update_every=1):
        # self.fig, self.ax = plt.subplots()
        self.simulation_state = simulation_state
        print(type(self.simulation_state))
        self.update_every = update_every
        
        self.initial_temperature = self.simulation_state.T
        self.initial_paircoupling = self.simulation_state.J
        self.initial_field = self.simulation_state.magnetic_field

        self._init_matplotlib()

    def _init_matplotlib(self):
        self.fig = plt.figure()
        self.ax = plt.imshow(self.simulation_state.state, cmap=cm.binary, animated=True)

        # Remove ticklabels from axes
        plt.tick_params(
                axis='both',          
                which='both',     
                bottom=False,      
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off
        
        plt.subplots_adjust(left=0.2, bottom=0.25)
        
        # Legend
        values = self.simulation_state.spinchoice
        colors = [self.ax.cmap(self.ax.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=values[i])) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        
        #Siders
        self.axspeed = self.fig.add_axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.speed_slider = Slider(self.axspeed, 'Speed', 1, self.simulation_state.columns*100, 
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

        # Reset button
        self.resetax = self.fig.add_axes([0.85, 0.25, 0.1, 0.04])
        self.reset_button = Button(self.resetax, 'Reset', color='red', hovercolor='0.975')

        # Bullet points to switch algorithm
        self.algorithm_ax = plt.axes([0.1, 0.5, 0.15, 0.15], facecolor='white')
        self.algoritm_menu = RadioButtons(self.algorithm_ax, ('Metropolis', 'Wolff', 'Heat Bath'), active=0)

        self.animation_handler = animation.FuncAnimation(self.fig, self.update, blit=True, interval=5)
        
        print('Windows created')
        print('Starting app')


    def update(self, *args):
        """
        Updates the figure.
        """
        # Refresh parameters that have changed via on screen input
        self.slider_changes()
        self.reset_button.on_clicked(self.reset)
        self.algoritm_menu.on_clicked(self.alogorithm_choice)
        
        for i in range(self.update_every):
            self.simulation_state.update_and_save()
        self.ax.set_array(self.simulation_state.state)
        return self.ax,

    def slider_changes(self):
        """
        Updates the values of parameters that changed via the sliders
        """
        self.simulation_state.steps_per_refresh = int(self.speed_slider.val)
        self.simulation_state.T = float(self.temp_slider.val)
        self.simulation_state.J = float(self.coupling_slider.val)
        self.simulation_state.magnetic_field = float(self.field_slider.val)


    def reset(self, event):
        """
        Resets all paramaters to their initial value
        """
        print(event)
        self.simulation_state.T = self.initial_temperature
        self.simulation_state.J = self.initial_paircoupling
        self.simulation_state.magnetic_field = self.initial_field
        self.temp_slider.reset()
        self.coupling_slider.reset()
        self.field_slider.reset()

    def alogorithm_choice(self, label):
        """
        Switches between different algorithms
        
        :param label (str): name of the algorithm that is choosen
        """
        self.simulation_state.algorithm_selected = label
        self.fig.canvas.draw_idle()