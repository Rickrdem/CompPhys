from dynamics import Dynamics
from viewer import Viewer
import matplotlib.pyplot as plt

def main():
    state = Dynamics(xsize=120, ysize=80)
    view = Viewer(state)
    plt.show()
    return state

if __name__ == '__main__':
    state = main()
