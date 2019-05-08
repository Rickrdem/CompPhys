import numpy as np
from numba import njit, jit

lattice_velocity = np.array([(x, y) for y in [-1, 0, 1] for x in [-1, 0, 1]], dtype=np.float)

lattice_weight = np.array([1 / 9 for i in range(9)])
lattice_weight[::2] = [1 / 36 for i in lattice_weight[::2]]
lattice_weight[4] = 4 / 9

VELOCITY_C = 1 / np.sqrt(3)  # sound velocity/ not sure why 2/sqrt(3) works but 1/sqrt(3) doesn't
TIME_C = 1

@njit()
def collision(flowin):
    # print("colliding")
    flowout = flowin.copy()
    xsize, ysize, connections = flowin.shape
    flow_eq = flowin.copy()*0
    for x in range(xsize):
        for y in range(ysize):
            # if mask[x,y]: continue
            density = 0
            velocity = np.array([0., 0.])
            for i in range(connections):
                density += flowin[x, y, i]
                velocity += flowin[x, y, i] * lattice_velocity[i]
            if density != 0: velocity /= density
            velocity/=2
            # else: velocity*=0
            for i in range(connections):
                product = lattice_velocity[i] @ velocity
                velocity_squared = velocity @ velocity
                flow_eq[x,y,i] = lattice_weight[i] * density * (
                        1+product / (VELOCITY_C ** 2) +
                        product**2 / (2 * VELOCITY_C ** 4) -
                        velocity_squared / (2 * VELOCITY_C ** 2)
                )
    flowout -= 1 / TIME_C * (flowin - flow_eq)
    return flowout

@njit()
def flow_equilibrium(velocity, density=1):
    xsize, ysize, dimensions = velocity.shape
    connections = 9
    flow_eq = np.empty((xsize, ysize, connections))

    for x in range(xsize):
        for y in range(ysize):
            for i in range(connections):
                product = lattice_velocity[i] @ velocity[x,y].T
                velocity_squared = velocity[x,y] @ velocity[x,y].T
                flow_eq[x, y ,i] = lattice_weight[i] * density * (
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
            if mask[x,y]:
                for i in range(connections):
                    flowout[x,y,8-i] = flowin[x,y,i]
    return flowout

@njit()
def streaming(flowout):
    # print('streaming')
    xsize, ysize, connections = flowout.shape
    flowin = np.empty_like(flowout)
    for x in range(xsize):
        for y in range(ysize):
            for i in range(connections):  # periodic boundarie
                x_neighbour = (x+int(lattice_velocity[i][0]))%xsize
                y_neighbour = (y+int(lattice_velocity[i][1]))%ysize
                flowin[int(x_neighbour), int(y_neighbour), i] = flowout[x, y, i]
                flowin[x_neighbour, y_neighbour, 4] = flowout[x_neighbour, y_neighbour, 4]
    return flowin

@jit()
def boundary_correction(flowin):
    # left wall equilibriate
    xsize, ysize, connections = flowin.shape

    density = np.sum(flowin[0,:,:], axis=-1)
    left_wall_velocity = np.empty((ysize, 2))
    left_wall_velocity[:,:] = np.sum(flowin[0,:,:,None] * lattice_velocity[None, :], axis=1)
    left_wall_velocity/=density[:,None]

    flow_eq = np.zeros((ysize, 9))

    for y in range(ysize):
        for i in range(connections):
            product = lattice_velocity[i] @ left_wall_velocity[y].T
            velocity_squared = left_wall_velocity[y] @ left_wall_velocity[y].T
            a = lattice_weight[i] * density[i] * (
                    1 + product / VELOCITY_C ** 2 +
                    product**2 / (2 * VELOCITY_C ** 4) -
                    velocity_squared / (2 * VELOCITY_C ** 2)
            )
            # print(a, product, velocity_squared, lattice_weight[i])
            flow_eq[y, i] = a

    flowin[0,:,0:3] = 0.001#flow_eq[:,0:3]

    #  right wall
    # flowin[-1, :, 6:9] = flowin[-2, :, 6:9]
    flowin[-2,:,:] = 0

    return flowin

# @njit()
def update(flowin, mask, steps=10):
    for step in range(steps):
        flowin = boundary_correction(flowin)  # not working for left wall
        #inflow # u[:, 0, :] = vel[:, 0, :]  # Left wall: compute density from known populations.
                # rho[0, :] = 1. / (1. - u[0, 0, :]) * (sumpop(fin[i2, 0, :]) + 2. * sumpop(fin[i1, 0, :]))

        flowout = collision(flowin)

        flowout = obstacle_flip(flowin, flowout, mask)

        flowin = streaming(flowout)
    density = np.sum(flowin, axis=-1)
    velocity = np.sum(flowin[:,:,:,None] * lattice_velocity[None,None,:], axis=2)/density[:,:,None]
    return flowin, velocity
