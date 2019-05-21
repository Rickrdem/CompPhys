from dynamics import Dynamics
from viewer import Viewer
import matplotlib.pyplot as plt

def main():
    state = Dynamics(xsize=520, ysize=180, show_every=20)
#    state = Dynamics(xsize=100, ysize=40, show_every=10)
    view = Viewer(state)
    plt.show()
    return state

if __name__ == '__main__':
    state = main()
