import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab   
import scipy.stats as stats

import functions as func


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
    
    p, cov = np.polyfit(x=time, y=game.diffusion, deg=1, full=False, cov=True)
   
    ax.plot(time, game.diffusion, c='b', label='$Diffusion$')
    ax.plot(time, p[0]*time+p[1], color='black', label='linear fit')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("$<(x(t)-x(0))^2>$")
    ax.set_title("Fit: $<(x(t)-x(0))^2>$=2Dt+b; D={D:.4f}$\pm$ {std:.4f}, b={b:.3f}".format(D=p[0]/2, std=np.sqrt(cov[0,0]), b=p[1]))
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
    time = np.arange(game.h, 2.15*game.h*len(game.kinetic_energy), 2.15*game.h)
    temp = np.array(game.temperature)
    
    # Set up the axes with gridspec
    fig = plt.figure()
    grid = plt.GridSpec(1, 6, fig, hspace=0.2, wspace=0.2)
    
    ax = fig.add_subplot(grid[0, 0:4])
    ax2 = fig.add_subplot(grid[0, 4:], sharey=ax)

    # Plot data in main axis ax
    ax.plot(time, temp[:,1], c='b', label='Measured')
    ax.plot(time, temp[:,0], c='r', label='Set')

    # Normalized histogram on the attached axis ax2    
    n, bins, patches = ax2.hist(temp[:,1], int(np.sqrt(len(temp[:,1]))), density=1,
                orientation='horizontal', color='blue')

    (mu, sigma) = stats.norm.fit(temp[:,1])
    y = mlab.normpdf(bins, mu, sigma)
    ax2.hlines(y=mu, xmin=0, xmax=np.max(y), color='red', linestyle=':')
    
    ax2.plot(y, bins, 'r--', linewidth=2, label='pdf')

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temparature")
    ax2.set_xlabel("Probability density")
    
    ax2.set_title("$\mu$, $\sigma$ = {m:.2f}, {s:.2f}".format(m=mu, s=sigma))
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    ax.legend()
    ax2.legend()
    fig.show()
    
    
def bootstrap(observable):
    n = 1000
    random_set = np.random.choice(observable, size=n*len(observable)).reshape((len(observable), n))
    set_of_averages = np.average(random_set, axis=0)
    sigma = np.sqrt(np.average(np.square(set_of_averages))-np.square(np.average(set_of_averages)))
    return sigma