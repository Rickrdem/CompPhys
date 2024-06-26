import numpy as np
from numba import njit, prange

LATTICE_VELOCITY = np.array([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]], dtype=np.float)

LATTICE_WEIGHT = np.array([1 / 9 for i in range(9)])
LATTICE_WEIGHT[::2] = [1 / 36 for i in LATTICE_WEIGHT[::2]]
LATTICE_WEIGHT[4] = 4 / 9


VELOCITY_C = 1 / np.sqrt(3)         # sound velocity/ not sure why 2/sqrt(3) works but 1/sqrt(3) doesn't

ULB = 0.04                          # Velocity in lattice units.
r = 180/9                            # Radius of obstacle
Re = 220.                          # Reynolds number
TIME_C = (3.*(ULB*r/Re)+0.5)  # Relaxation parameter.

@njit()
def initial_velocity(x, y):
    # d = np.array([0,1])
    # return 1/100*(d[None, None, :]-1)*(0.1+0.01*np.sin(y[None, :, None]/30)) + 0*x[:, None, None]
    velocity = np.empty((len(x), len(y), 2))
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            for d in range(2):
                velocity[i, j, d] = 0.04*(1-d)*(1+1e-4*np.sin(y_val/179*2*np.pi))
    return velocity
    # def func(x,y,d):
    #     return 1/100*(d-1)*(0.1+0.01*np.sin(y/30))
    # return np.fromfunction(func, (len(x), len(y), 2))


@njit()
def influx(y):
    """
    Incoming fluid stream at left boundary
    :param y: y-coordinate at the left boundary
    :return: fluid velocity
    """
    velocity = np.empty((len(y), 2))
    for j, y_val in enumerate(y):
        for d in range(2):
            velocity[j, d] = 0.04*(1-d)*(1+1e-4*np.sin(y_val/179*2*np.pi))
    return velocity


@njit(parallel=True)
def macroscopic_parameters(flowin):
    xsize, ysize, connections = flowin.shape
    velocity = np.zeros((xsize, ysize, 2))
    density = np.zeros((xsize, ysize))
    for x in prange(xsize):
        for y in range(ysize):
            for i in range(connections):
                delta_density = flowin[x, y, i]
                density[x, y] += delta_density
                velocity[x, y] += flowin[x, y, i] * LATTICE_VELOCITY[i]
            velocity[x,y] /= density[x,y]
    return velocity, density

@njit()
def collision(flowin, floweq):
    # print("colliding")
    # flowout = flowin.copy()
    flowout = flowin - 1 / TIME_C * (flowin - floweq)
    return flowout

def equilibrium_source(rho,u):              # Equilibrium distribution function.
    xshape, yshape, _ = u.shape
    cu   = 3.0 * np.dot(LATTICE_VELOCITY,u.transpose(0,2,1)) # i x y -> x y i
    # cu = 3.0 * np.dot()
    usqr = 3./2.*(u[:,:,0]**2+u[:,:,1]**2)
    feq = np.zeros((xshape,yshape,9))
    for i in range(9): feq[:,:,i] = rho*LATTICE_WEIGHT[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq

@njit(parallel=True)
def flow_equilibrium(velocity, density):
    xsize, ysize, dimensions = velocity.shape
    connections = 9
    flow_eq = np.empty((xsize, ysize, connections))

    for x in prange(xsize):
        for y in prange(ysize):
            velocity_squared = velocity[x, y] @ velocity[x, y].T
            for i in prange(connections):
                product = LATTICE_VELOCITY[i] @ velocity[x, y].T
                # flow_eq[x, y, i] = LATTICE_WEIGHT[i] * density[x, y] * (
                #         1 + product / VELOCITY_C ** 2 +
                #         np.square(product) / (2 * VELOCITY_C ** 4) -
                #         velocity_squared / (2 * VELOCITY_C ** 2)
                # )
                flow_eq[x, y, i] = LATTICE_WEIGHT[i] * density[x, y] * (
                    1 + 3*product + #/ VELOCITY_C +
                    9/2 * product**2 -#/VELOCITY_C**2 -
                    3/2*velocity_squared #/VELOCITY_C**2
                )
    return flow_eq


@njit(parallel=True)
def obstacle_flip(flowin, flowout, mask):
    xsize, ysize, connections = flowin.shape
    for x in prange(xsize):
        for y in prange(ysize):
            if mask[x, y]:
                for i in prange(connections):
                    flowout[x, y, i] = flowin[x, y, 8-i]
    return flowout


@njit(parallel=True)
def streaming(flowout):
    # print('streaming')
    xsize, ysize, connections = flowout.shape
    flowin = np.zeros_like(flowout)
    for x in prange(xsize):
        for y in prange(ysize):
            for i in prange(connections):  # periodic boundaries
                x_neighbour = (x + int(LATTICE_VELOCITY[i, 0])) %xsize
                y_neighbour = (y + int(LATTICE_VELOCITY[i, 1])) %ysize
                # if xsize > x_neighbour >= 0 and ysize > y_neighbour >= 0:
                    # flowin[x, y, i] = flowout[x_neighbour, y_neighbour, i]
                    # flowin[x, y, 4] = flowout[x, y, 4]
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
        flowin[-1, :, 6:] = flowin[-2, :, 6:]  # outflow on the right wall
        #
        velocity, density = macroscopic_parameters(flowin)
        # velocity[:, :, 0] /= density[:, :]
        # velocity[:, :, 1] /= density[:, :]

        # left wall #############
        xsize, ysize, connections = flowin.shape
        velocity[0, :, :] = influx(np.arange(ysize))
        # density[0, :] = (np.sum(flowin[0, :, 3:6], axis=-1) + 2. * np.sum(flowin[0, :, 6:], axis=-1)) / (1. - velocity[0, :, 0])
        # density[0, :] = np.sum(flowin[0,:,3:6], axis=-1) + 2* np.sum(flowin[0,:,6:], axis=-1)
        ##############

        floweq = flow_equilibrium(velocity, density)
        # floweq = equilibrium_source(density, velocity)

        flowin[0, :, :3] = floweq[0, :, :3]

        flowout = collision(flowin, floweq)
        # flowout = flowin

        flowout = obstacle_flip(flowin, flowout, mask)

        flowin = streaming(flowout)
        # for i in range(9):
        #     flowin[:,:,i] = np.roll(np.roll(flowout[:,:,i], int(LATTICE_VELOCITY[i,0]), axis=0), int(LATTICE_VELOCITY[i,1]), axis=1)

    # density = np.sum(flowin, axis=-1)
    # velocity = np.sum(flowin[:, :, :, None] * LATTICE_VELOCITY[None, None, :], axis=2) / density[:, :, None]
    return flowin, velocity
