import numpy as np
from numba import njit, jit

LATTICE_VELOCITY = np.array([(x, y) for y in [-1, 0, 1] for x in [-1, 0, 1]], dtype=np.float)

LATTICE_WEIGHT = np.array([1 / 9 for i in range(9)])
LATTICE_WEIGHT[::2] = [1 / 36 for i in LATTICE_WEIGHT[::2]]
LATTICE_WEIGHT[4] = 4 / 9

VELOCITY_C = 1 / np.sqrt(3)  # sound velocity/ not sure why 2/sqrt(3) works but 1/sqrt(3) doesn't
TIME_C = 0.5

def initial_velocity(x, y):
    return np.fromfunction(lambda x,y,d:(d-1)*(0.01+0.0001*np.sin(x/10)), (len(x), len(y), 2))

@njit()
def macroscopic_parameters(flowin):
    xsize, ysize, connections = flowin.shape
    velocity = np.zeros((xsize, ysize, 2))
    density = np.zeros((xsize,ysize))
    for x in range(xsize):
        for y in range(ysize):
            for i in range(connections):
                delta_density = flowin[x, y, i]
                density[x,y] += delta_density
                velocity[x,y] += flowin[x, y, i]*LATTICE_VELOCITY[i]#/delta_density
            # velocity[x,y] /= density[x,y]
    # velocity /= density
    return velocity, density

@njit()
def collision(flowin, floweq):
    # print("colliding")
    # flowout = flowin.copy()
    flowout = flowin - 1 / TIME_C * (flowin - floweq)
    return flowout


@jit()
def flow_equilibrium(velocity, density=1):
    xsize, ysize, dimensions = velocity.shape
    connections = 9
    flow_eq = np.empty((xsize, ysize, connections))

    for x in range(xsize):
        for y in range(ysize):
            for i in range(connections):
                if type(velocity) is type(density):
                    d = density[x,y]
                else:
                    d = density
                product = LATTICE_VELOCITY[i] @ velocity[x, y].T
                velocity_squared = velocity[x, y] @ velocity[x, y].T
                flow_eq[x, y, i] = LATTICE_WEIGHT[i] * d * (
                        1 + product / VELOCITY_C ** 2 +
                        np.square(product) / (2 * VELOCITY_C ** 4) -
                        velocity_squared / (2 * VELOCITY_C ** 2)
                )
    return flow_eq


@njit()
def obstacle_flip(flowin, flowout, mask):
    xsize, ysize, connections = flowin.shape
    for x in range(xsize):
        for y in range(ysize):
            if mask[x, y]:
                for i in range(connections):
                    flowout[x, y, 8 - i] = flowin[x, y, i]
    return flowout


@njit()
def streaming(flowout):
    # print('streaming')
    xsize, ysize, connections = flowout.shape
    flowin = np.zeros_like(flowout)
    for x in range(xsize):
        for y in range(ysize):
            for i in range(connections):  # periodic boundarie
                x_neighbour = (x + int(LATTICE_VELOCITY[i][0])) %xsize
                y_neighbour = (y + int(LATTICE_VELOCITY[i][1])) %ysize
                if xsize > x_neighbour >= 0:
                    if ysize > y_neighbour >= 0:
                        flowin[int(x_neighbour), int(y_neighbour), i] = flowout[x, y, i]
                        flowin[x_neighbour, y_neighbour, 4] = flowout[x_neighbour, y_neighbour, 4]
    return flowin


# @jit()
# def boundary_effect(flowin, velocity, density, floweq):
#     # left wall equilibriate
#     xsize, ysize, connections = flowin.shape
#
#     # density = np.sum(flowin[0, :, :], axis=-1)
#     # left_wall_velocity = np.empty((ysize, 2))
#     # left_wall_velocity[:, :] = np.sum(flowin[0, :, :, None] * LATTICE_VELOCITY[None, :], axis=1)
#     # left_wall_velocity /= density[:, None]
#     # left_wall_velocity[density!=0] = left_wall_velocity[density!=0]/density[density!=0,None]
#
#     flow_eq = np.zeros((ysize, 9))
#
#     for y in range(ysize):
#         for i in range(connections):
#             product = LATTICE_VELOCITY[i] @ velocity[0,y].T
#             velocity_squared = velocity[0,y] @ velocity[0,y].T
#             a = LATTICE_WEIGHT[i] * density[0,y] * (
#                     1 + product / VELOCITY_C ** 2 +
#                     product ** 2 / (2 * VELOCITY_C ** 4) -
#                     velocity_squared / (2 * VELOCITY_C ** 2)
#             )
#             flow_eq[0, y, i] = a
#
#     flowin[0, :, 0:3] = flow_eq[0, :,0:3]
#
#     #  right wall
#     flowin[-1, :, 6:9] = flowin[-2, :, 6:9]
#     # flowin[-2, :, :] = 0
#
#     return flowin


# @njit()
def update(flowin, mask, steps=10):
    for step in range(steps):
        flowin[-1, :, 6:9] = flowin[-2, :, 6:9]  # outflow on the right wall
        #
        velocity,density = macroscopic_parameters(flowin)
        velocity/=density[:,:,None]
        # left wall #############
        xsize, ysize, connections = flowin.shape
        velocity[0, :, :] = initial_velocity(np.arange(1), np.arange(ysize))
        density[0, :] = 1. / (1. - velocity[0, :, 0]) * (np.sum(flowin[0, :, 3:6], axis=-1) + 2. * np.sum(flowin[0, :, 0:3], axis=-1))
        ##############

        floweq = flow_equilibrium(velocity, density)

        # flowin = boundary_effect(flowin, velocity, density, floweq)  # not working for left wall
        # inflow # u[:, 0, :] = vel[:, 0, :]  # Left wall: compute density from known populations.
        # rho[0, :] = 1. / (1. - u[0, 0, :]) * (sumpop(fin[i2, 0, :]) + 2. * sumpop(fin[i1, 0, :]))

        flowout = collision(flowin, floweq)
        # flowout = flowin

        flowout = obstacle_flip(flowin, flowout, mask)

        flowin = streaming(flowout)

    # density = np.sum(flowin, axis=-1)
    # velocity = np.sum(flowin[:, :, :, None] * LATTICE_VELOCITY[None, None, :], axis=2) / density[:, :, None]
    return flowin, velocity
