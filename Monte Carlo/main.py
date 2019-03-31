import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dynamics import Dynamics

if __name__ == "__main__":
    plt.close("all")

    fig, ax = plt.subplots()
    simulation_state = Dynamics(fig, ax)
            
    # pass a generator in "emitter" to produce data for the update func
    ani = animation.FuncAnimation(fig, simulation_state.update, 1, interval=1, blit=True)
        
    fig.show()
    plt.show()