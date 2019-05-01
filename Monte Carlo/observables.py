"""
@author: Lennard Kwakernaak
@author: Rick Rodrigues de Mercado
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import matplotlib.mlab as mlab
import scipy.stats as stats

def trim_data(simulation_state, start, end=None):
    """
    Trim the data to remove a set of time steps.
    :param simulation_state: The state to trim in time
    :param start: The number of states to trim off of the front of the dataset.
    :param end: The number of states to trim off of the back of the dataset.
    :return: Trimmed simulation state.
    """
    simulation_state.energy = simulation_state.energy[start:end]
    simulation_state.magnetization = simulation_state.magnetization[start:end]
    return simulation_state

def magnetization(simulation_state):
    """
    Plots the average magnetization of the system. 
    """
    magnetization = simulation_state.magnetization
    average_magnetization = np.average(magnetization)
    magnetization_error = bootstrap(magnetization, np.average)
    time = np.arange(0, len(magnetization), 1)
    
    # Set up the axes with gridspec
    fig = plt.figure()
    grid = plt.GridSpec(1, 6, fig, hspace=0.2, wspace=0.2)

    ax = fig.add_subplot(grid[0, 0:4])
    ax2 = fig.add_subplot(grid[0, 4:], sharey=ax)

    # Plot data in main axis ax
    ax.plot(time, magnetization, c='b', label='Measured')

    # Normalized histogram on the attached axis ax2    
    n, bins, patches = ax2.hist(magnetization, int(np.sqrt(len(magnetization))), density=1,
                                orientation='horizontal', color='blue')

    (mu, sigma) = stats.norm.fit(magnetization)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.hlines(y=mu, xmin=0, xmax=np.max(y), color='red', linestyle=':')

    ax2.plot(y, bins, 'r--', linewidth=2, label='pdf')

    ax.set_xlabel("Time steps")
    ax.set_ylabel(r"$m$")
    ax2.set_xlabel("Probability density")

    ax2.set_title("$\mu$={m:.2f}, $\sigma$={s:.3f}, err={e:.4f}".format(m=mu, 
                                                                      s=sigma,
                                                                      e=magnetization_error))
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax.legend()
    ax2.legend()
    fig.show()

    
    return average_magnetization, magnetization_error
    
def susceptibility(simulation_state):
    """
    Calculates the susceptibility per spin.
    """
    func = lambda x: np.var(x)*simulation_state.num_spins/simulation_state.temp
    sus = func(simulation_state.magnetization)
    sus_error = bootstrap(simulation_state.magnetization, func)
    return sus, sus_error

def specific_heat(simulation_state):
    """
    Calculates the specific heat of the system.
    """
    func = lambda x: np.var(x)/simulation_state.num_spins/(simulation_state.temp**2)
    s_heat = func(simulation_state.energy)
    s_heat_err = bootstrap(simulation_state.energy, func)

    return s_heat, s_heat_err

def magn_vs_field(simulation_state, temp, fig, ax):
    magnetization = simulation_state.magnetization
    field = np.arange(-5, 5, 0.1)
    
    ax.plot(field, magnetization, color='blue', label="T="+str(temp))
    
    ax.set_xlabel("H")
    ax.set_ylabel("m")
    ax.legend()


def nn_interaction(shape):
    middle = len(shape)//2
    m = shape[middle]
    shape[middle] = 0
    return np.sum(m*shape)

def energy(state, j_coupling=1, H=0):
    """ 
    Hamiltonian operator on a state vector for free bounds
    """
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    return -j_coupling * np.sum(ndi.generic_filter(
        state, function=nn_interaction, footprint=footprint, mode='wrap'))

def bootstrap(data, function, n=100):
    """
    Bootstrap an observable, given a specific function
    """
    observables = []
    for i in range(n):
        resampled = np.random.choice(data, size=len(data), replace=True)
        observables.append(function(resampled))
    return np.std(observables)

