# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:47:16 2019

@author: rmerc
"""
import numpy as np
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots()
    ax.hist(func.abs(game.velocities), bins=int(game.particles**0.5))
    ax.set_xlabel("Velocity")
    ax.set_ylabel("#")
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

def pair_correlation(game, fig, ax):
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
    fig, ax = plt.subplots()
    time = np.arange(game.h, 2.15*game.h*len(game.kinetic_energy), 2.15*game.h)
    ax.plot(time, np.array(game.temperature)[:,0], c='r', label='Set')
    ax.plot(time, np.array(game.temperature)[:,1], c='b', label='Measured')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temparature")
    ax.legend()
    fig.tight_layout()
    fig.show()