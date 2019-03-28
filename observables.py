import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab   
import scipy.stats as stats

import functions as func


def energy(simulation_state):
    """Timetrace of the energy in the system. Kinetic energy, potential enery 
    and the sum of those is displayed"""
    fig, ax = plt.subplots()
    time = np.arange(0, simulation_state.h * len(simulation_state.kinetic_energy), simulation_state.h)
    total_energy = np.asarray(simulation_state.kinetic_energy) + simulation_state.potential_energy
    ax.plot(time, np.array(simulation_state.kinetic_energy[:]), c='r', label='$E_k$')
    ax.plot(time, simulation_state.potential_energy, c='b', label='$E_p$')
    ax.plot(time, total_energy, c='black', label='$E_t$')
    
    kinetic_average = np.average(simulation_state.kinetic_energy)
    kinetic_sigma = bootstrap(simulation_state.kinetic_energy, np.average)
    potential_average = np.average(simulation_state.potential_energy)
    potential_sigma = bootstrap(simulation_state.potential_energy, np.average)
    tot_average = np.average(total_energy)
    tot_sigma = bootstrap(total_energy, np.average)
    
    ax.set_title(r"""$\mu(E_k, E_p, E_t)$ = ({a:.1f}, {b:.1f}, {c:.1f}),
                 $\sigma(E_k, E_p, E_t)$ = ({d:.1f}, {e:.1f}, {f:.1f})""".format(a=kinetic_average,
                                                                                  b=potential_average,
                                                                                  c=tot_average,
                                                                                  d=kinetic_sigma,
                                                                                  e=potential_sigma,
                                                                                  f=tot_sigma))

    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$E/\epsilon$")
    ax.legend()
    fig.tight_layout()
    fig.show()

def velocity_distribution(game):
    """Histogram of velicity distribution overlayed with a maxwell-boltzmann pfd fit"""
    fig, ax = plt.subplots()

    velocities = func.abs(game.velocities)
        
    y, bin_edges = np.histogram(velocities, bins=int(np.sqrt(len(velocities))), density=True)
    bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
    menStd     = np.sqrt(y)/np.sum(y)
    width      = np.max(velocities)/int(np.sqrt(len(velocities)))
    ax.bar(bincenters, y, width=width, color='b', yerr=menStd)
    
    maxwell = stats.maxwell
    params = maxwell.fit(velocities, floc=0)
    mean, var = maxwell.stats(*params, moments='mv')
    x = np.linspace(0, np.max(velocities), 100)
    ax.plot(x, maxwell.pdf(x, *params), color='black', lw=3)   
    
    ax.set_xlabel("Velocity")
    ax.set_ylabel("#/N")
    ax.set_title(r"$\mu$={m:.2f}, $\sigma$={s:.3f}".format(m=mean, s=var*.5))
    fig.tight_layout()
    fig.show()

def diffusion(simulation_state):
    """Timetrace of the diffusion in the system."""
    fig, ax = plt.subplots()
    time = np.arange(0, simulation_state.h * len(simulation_state.diffusion), simulation_state.h)
    
    p, cov = np.polyfit(x=time, y=simulation_state.diffusion, deg=1, full=False, cov=True)
   
    ax.plot(time, simulation_state.diffusion, c='b', label='$Diffusion$')
    ax.plot(time, p[0]*time+p[1], color='black', label='linear fit')
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$<(r(t)-r(0))^2>$")
    ax.set_title("Linear fit: y=ax+b; a={a:.4f}$\pm$ {std:.4f}, b={b:.3f}".format(a=p[0], std=np.sqrt(cov[0,0]), b=p[1]))
    ax.legend()
    fig.tight_layout()
    fig.show()

def pair_correlation(simulation_state, fig=None, ax=None):
    """Plot of the pair correlation function"""
    dr = 1 / simulation_state.particles ** (1 / 2)
    radius = np.arange(0.4, simulation_state.size[0], dr)
    pair_correlation = []
    for r in radius:
        distances = np.sqrt(np.sum(np.square(simulation_state.distances), axis=2))
        X = (distances > r) == True
        Y = (distances < r+dr) == True
        Z = X==Y
        n_r = np.sum(Z, axis=1)
        g_r = 2 * simulation_state.volume / (simulation_state.particles * (simulation_state.particles - 1)) * np.average(n_r) / (4 * np.pi * r * r * dr)
        pair_correlation.append(g_r)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(radius, pair_correlation/(np.sum(pair_correlation)*dr), c='b', label='Pair Correlation')
    ax.set_xlabel(r"r/$\sigma$")
    ax.set_ylabel(r"g(r/$\sigma$)")
    ax.legend()
    fig.tight_layout()
    fig.show()
  
def temperature(simulation_state):
    """Timetrace of the temperature of the system."""
    time = np.arange(0, simulation_state.h * len(simulation_state.kinetic_energy), simulation_state.h)
    temp = np.array(simulation_state.temperature)
    set_temp = np.array(simulation_state.set_temperature)
    
    # Set up the axes with gridspec
    fig = plt.figure()
    grid = plt.GridSpec(1, 6, fig, hspace=0.2, wspace=0.2)
    
    ax = fig.add_subplot(grid[0, 0:4])
    ax2 = fig.add_subplot(grid[0, 4:], sharey=ax)

    # Plot data in main axis ax
    ax.plot(time, temp, c='b', label='Measured')
    ax.plot(time, set_temp, c='r', label='Set')

    # Normalized histogram on the attached axis ax2    
    n, bins, patches = ax2.hist(temp, int(np.sqrt(len(temp))), density=1,
                orientation='horizontal', color='blue')

    (mu, sigma) = stats.norm.fit(temp)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.hlines(y=mu, xmin=0, xmax=np.max(y), color='red', linestyle=':')
    
    ax2.plot(y, bins, 'r--', linewidth=2, label='pdf')

    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$T$")
    ax2.set_xlabel("Probability density")
    
    ax2.set_title("$\mu$={m:.2f}, $\sigma$={s:.3f}".format(m=mu, s=sigma))
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    ax.legend()
    ax2.legend()
    fig.show()
    
def specific_heat(simulation_state):
    K = np.array(simulation_state.kinetic_energy)
    N = simulation_state.particles
    a = np.average(np.square(K))
    b = np.square(np.average(K))
    
    c_v = 1/(2/(3*N)*(1-((a/b)-1)*3*N/2))
    return c_v

def bootstrap(data, function, n=100):
    observables = []
    for i in range(n):
        resampled = np.random.choice(data, size=len(data), replace=True)
        observables.append(function(resampled))
    return np.std(observables)
