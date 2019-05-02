from dynamics import Dynamics
from viewer import Viewer

def main():
    state = Dynamics()
    view = Viewer(state)
    return state

if __name__ == '__main__':
    state = main()
