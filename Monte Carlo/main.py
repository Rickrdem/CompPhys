import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

import observables as obs
from dynamics import Dynamics
from viewer import Viewer


if __name__ == "__main__":
    plt.close("all")

    
    J = 1
    T = 2.26
    
    simulation_state = Dynamics(J, T)
            
    # pass a generator in "emitter" to produce data for the update func
    # ani = animation.FuncAnimation(fig, simulation_state.update, 1, interval=1, blit=True)

    viewport = Viewer(simulation_state)

    # fig.show()
    plt.show()
        
    print("""
        Pair-pair coupling: J = {j}
        Temperature: T = {t}
        """.format(j=J,
                    t=T))

    obs.magnetization(simulation_state)
    obs.susceptibility(simulation_state)
    
    print(simulation_state.m)
    print("Done")