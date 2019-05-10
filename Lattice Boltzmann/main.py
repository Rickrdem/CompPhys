from dynamics import Dynamics
from viewer import Viewer
import matplotlib.pyplot as plt

def main():
    state = Dynamics(xsize=400)
    view = Viewer(state)
    plt.show()
    return state

if __name__ == '__main__':
    state = main()
