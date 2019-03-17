import numpy as np
import matplotlib.pyplot as plt
import functions as func

import scipy.stats as stats

def energy(game):
    fig, ax = plt.subplots()
    time = np.arange(game.h, 2.15*game.h*len(game.kinetic_energy), 2.15*game.h)
    ax.plot(time, np.array(game.kinetic_energy[:]),c='r', label='$E_{kin}$')
    ax.plot(time, game.potential_energy, c='b', label='$E_{pot}$')
    ax.plot(time, np.asarray(game.kinetic_energy)+game.potential_energy, c='black', label='H')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy ($1/\epsilon$)")
    ax.legend()
    fig.tight_layout()
    fig.show()

def velocity_distribution(game):
    """Histogram of velicity distribution overlayed with a maxwell-boltzmann pfd fit"""
    fig, ax = plt.subplots()

    velocities = func.abs(game.velocities)
    
    maxwell = stats.maxwell
    params = maxwell.fit(velocities, floc=0)
    x = np.linspace(0, np.max(velocities), 100)
    ax.hist(velocities, bins=int(game.particles**(1/2)), density=1, facecolor='blue')
    ax.plot(x, maxwell.pdf(x, *params), color='black', lw=3)   

    ax.set_xlabel("Velocity")
    ax.set_ylabel("#/N")
    ax.set_title("(params)="+str(params))
    fig.tight_layout()
    fig.show()

def diffusion(game):
    fig, ax = plt.subplots()
    time = np.arange(game.h, 2.15*game.h*len(game.diffusion), 2.15*game.h)
    ax.plot(time, game.diffusion, c='b', label='$Diffusion$')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("$<(x(t)-x(0))^2>$")
    ax.legend()
    fig.tight_layout()
    fig.show()

def pair_correlation(game, fig=None, ax=None):
    dr = 1/game.particles**(1/2)
    radius = np.arange(0.4, game.size[0], dr)
    pair_correlation = []
    for r in radius:
        distances = np.sqrt(np.sum(np.square(game.distances), axis=2))
        X = (distances > r) == True
        Y = (distances < r+dr) == True
        Z = X==Y
        n_r = np.sum(Z, axis=1)
        g_r = 2*game.volume/(game.particles*(game.particles-1))*np.average(n_r)/(4*np.pi*r*r*dr)
        pair_correlation.append(g_r)
      
#    fig, ax = plt.subplots()
    ax.plot(radius, pair_correlation/(np.sum(pair_correlation)*dr), c='b', label='Pair Correlation')
    ax.set_xlabel("r/$\sigma$")
    ax.set_ylabel("g(r/$\sigma$)")
    ax.legend()
    fig.tight_layout()
    fig.show()
  
def temperature(game):
    fig, [ax, ax1] = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
    time = np.arange(game.h, 2.15*game.h*len(game.kinetic_energy), 2.15*game.h)
    temp = np.array(game.temperature)
    
    mu = np.average(temp[:,1])
    sigma = np.sqrt(np.average(np.square(temp[:,1])) - np.square(np.average(temp[:,1])))

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    ax.plot(time, temp[:,0], c='r', label='Set')
    ax.plot(time, temp[:,1], c='b', label='Measured')
    
    ax1.plot(stats.norm.pdf(x, mu, sigma), x, color='blue', label='Temperature pdf')    
    ax1.hlines(y=mu, xmin=0, xmax=np.max(stats.norm.pdf(x, mu, sigma)), linestyle=':')
    
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temparature")
    ax1.set_xlabel("Probability")
    
    ax1.set_title("mean, $\sigma$ = {m:.2f}, {s:.2f}".format(m=mu, s=sigma))
    
    ax.legend()
    ax1.legend()
    fig.tight_layout()
    fig.show()