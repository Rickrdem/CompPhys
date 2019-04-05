import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import observables as obs
from dynamics import Dynamics


if __name__ == "__main__":
    plt.close("all")

    fig, ax = plt.subplots()
    
    J = -10000
    T = 1
    
    simulation_state = Dynamics(J, T, fig, ax)
            
    # pass a generator in "emitter" to produce data for the update func
    ani = animation.FuncAnimation(fig, simulation_state.update, 1, interval=1, blit=True)

    fig.show()
    plt.show()
        
    print("""
        J = {j}
        T = {t}
        """.format(j=J,
                    t=T))

    obs.magnetization(simulation_state)
    obs.susceptibility(simulation_state)
    
    print(simulation_state.m)
    print("Done")