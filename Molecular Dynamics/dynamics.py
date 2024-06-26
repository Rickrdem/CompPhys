"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""

import numpy as np
import functions as func
import observables as obs

class Dynamics():
    """
    Molecular dynamics integrator.

    This class contains the methods required to simulate a molecular dynamics simulation.
    Typical usage is to initialise with a state and propagate that state with the update method.

    :param state: A Nxd numpy array containing the starting positions of all particles.
    :param h: The time step in reduced units that is used in Verlet integration.
    :param T: The temperature in reduced units that the program is set to.
    :param size: A tuple of length d that contains the size size of the periodic bounds.
    :param dtype: The datatype class that is used for the simulation.
    """
    def __init__(self, state, h=0.005, T=0.5, size=(10, 10, 10), dtype=np.float32):
        self.h = h
        self.T = T
        self.time = 0

        self.size = size
        self.particles = np.shape(state)[0]
        self.dimensions = np.shape(state)[1]

        self.volume = self.size[0] * self.size[1] * self.size[2]
        self.density = self.particles / self.volume
        self.measured_temperature = 0
        self.thermostat = True

        self.dtype = dtype

        self.positions = np.zeros([self.particles, self.dimensions], dtype=dtype)
        self.original_positions = np.zeros([self.particles, self.dimensions], dtype=dtype)
        self.positions_not_bounded = np.zeros([self.particles, self.dimensions], dtype=dtype)
        self.velocities = np.zeros([self.particles, self.dimensions], dtype=dtype)
        self.forces = np.zeros([self.particles, self.dimensions], dtype=dtype)
        self.distances = np.zeros([self.particles, self.particles, self.dimensions], dtype=dtype)

        self.potential_energy = []
        self.kinetic_energy = []
        self.diffusion = []
        self.temperature = []
        self.set_temperature = []

        self.generate_state(state)

    def generate_state(self, state):
        """ 
        Loads up the original state and adds some noise in the velocities.
        """
        self.positions[:] = np.asarray(state, dtype=self.dtype).copy()
        self.original_positions = self.positions.copy()
        self.positions_not_bounded = state.copy()
        self.velocities[:] = np.random.normal(0, 0.1, size=(self.particles, self.dimensions))
        
        #No net velocity in the system
        self.velocities[:] = self.velocities - np.average(self.velocities, axis=0)[None, :] 
        
        self.measured_temperature = np.average(np.square(self.velocities))

    def update(self, a):
        """ 
        Updates the dynamic state

        Moves all the particles around
        """

        self.velocities_update()  # first half

        if self.thermostat > 0:
            lambda_ = np.sqrt(((self.particles - 1) * 3 * self.T) / (np.sum(np.square(self.velocities))))
            self.velocities = lambda_ * self.velocities
            if abs(self.T - self.measured_temperature) < 0.001:
                self.thermostat = False

        self.positions_update()
        self.distances_update()
        self.forces_update()
        self.velocities_update()  # second half

        self.measured_temperature = obs.temperature(func.abs(self.velocities))

        self.positions[:, :] %= np.asarray(self.size)[np.newaxis, :]

        self.kinetic_energy.append(obs.energy(func.abs(self.velocities)))
        self.potential_energy.append(func.sum_potential_jit(self.distances))
        self.diffusion.append(np.average(np.square((self.positions_not_bounded - self.original_positions))))
        self.temperature.append(self.measured_temperature)
        self.set_temperature.append(self.T)

        self.time += self.h
        
    def positions_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.positions[:] = self.positions + self.h * self.velocities
        self.positions_not_bounded[:] = self.positions_not_bounded + self.h * self.velocities

    def velocities_update(self):
        """Update the positions of all particles in-place using velocity-verlet"""
        self.velocities[:] = self.velocities + self.h / 2 * (self.forces)

    def forces_update(self):
        """Update the forces acting on all particles in-place using velocity-verlet"""
        self.forces[:] = np.sum(func.force_jit(self.distances), axis=1)

    def distances_update(self):
        """Update the NxNxD distance matrix with the nearest image convention"""
        self.distances[:] = func.distance_jit(self.positions, self.size)
