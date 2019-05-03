import numpy as np
from numba import njit

lattice_velocity = np.array([(x, y) for y in [-1, 0, 1] for x in [-1, 0, 1]], dtype=np.float)

lattice_weight = np.array([1 / 9 for i in range(9)])
lattice_weight[::2] = [1 / 36 for i in lattice_weight[::2]]
lattice_weight[4] = 4 / 9

VELOCITY_C = 1 / np.sqrt(3)
TIME_C = 1

@njit()
def collision(flow):
    flow = flow.copy()
    density = 0
    velocity = np.array([0.,0.])
    for i in range(9):
        density += flow[i]
        velocity += flow[i]*lattice_velocity[i]
    velocity/=density

    flow_eq = []
    for i in range(9):
        product = lattice_velocity[i] @ velocity.T
        velocity_squared = velocity @ velocity.T
        flow_eq.append(
            lattice_weight[i]*density*(
                1 + product / VELOCITY_C ** 2 + \
                np.square(product) / (2 * VELOCITY_C ** 4) -\
                velocity_squared/(2 * VELOCITY_C**2)
            )
        )
        flow[i] -= 1/TIME_C * (flow[i] - flow_eq[i])
    return flow

def streaming(content, neighbours):
    pass