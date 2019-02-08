# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:47:44 2018

@author: Lennard
"""

import numpy as np
import pyglet
import numba
import matplotlib.pyplot as plt

class gamestate():    
    def randomise_state(self):
        self.positions = np.random.randn(self.boids, 2)
        self.positions[:,0] *= self.size[0]
        self.positions[:,1] *= self.size[1]
        self.directions = np.random.uniform(0,2*np.pi,self.boids)#np.random.rand(self.boids)*np.pi%(2*np.pi)
        print(np.sum(np.exp(1j*self.directions)))
        print(circular_mean(self.directions)/np.pi)
#        print("boids",len(self.directions))
#        plt.hist(self.directions, bins=int(np.sqrt(len(self.directions))))
    
    def __init__(self, boids=2000, size=(500,500), eta=0.3, radius=10,
                 drawevery=1):
        self.boids = boids
        self.size = size
        self.eta = eta
        self.radius = radius
        self.randomise_state()
        self.drawevery = drawevery
        self.polarisation = []
        self.mean = []
        plt.hist(self.directions, int(np.sqrt(len(self.directions))))
        plt.show()
        
    def update(self, a):
        for step in range(self.drawevery):
            
#            noise = (np.random.rand(self.boids)-0.5) *2* np.pi
            noise = np.random.uniform(-np.pi,np.pi,self.boids)
            
            self.directions = average_circular_directions(self.positions,
                                                 self.directions,
                                                 self.radius)#*(1-self.eta)
            self.directions += noise * self.eta
            
            
            self.positions += np.array([np.cos(self.directions),
                                        np.sin(self.directions)]).T
            self.positions[:,0] %= self.size[0]
            self.positions[:,1] %= self.size[1]
            
            self.mean.append(circular_mean(self.directions))
            self.polarisation.append(polarisation(self.directions))
            self.directions %= (2*np.pi)

class viewport(pyglet.window.Window):
    def __init__(self, gamestate=None):
        assert gamestate != None
        super().__init__()
        self.gamestate = gamestate
#        self.label = pyglet.text.Label('Hello, world!')
        self.fpsdisplay = pyglet.window.FPSDisplay(self)

    def on_draw(self):
        self.clear()
        
        gamestate = self.gamestate
        x,y = gamestate.positions.T
        length = min(gamestate.radius,13)
        windowsize = self.get_size()
        x = windowsize[0]*x/gamestate.size[0]
        y = windowsize[1]*y/gamestate.size[1]
        dx, dy = np.cos(gamestate.directions)*length, np.sin(gamestate.directions)*length
        batch = pyglet.graphics.Batch()
        
        lines = np.dstack((x,y,x+dx,y+dy)).flatten()
        batch.add(2*len(x), pyglet.gl.GL_LINES, None,
                  ('v2f',
                   lines))
#        for boid in range(gamestate.boids):
#            batch.add(2, pyglet.gl.GL_LINES, None,
#                     ('v2f',
#                     (x[boid], y[boid], x[boid]+dx[boid], y[boid]+dy[boid])))
        batch.draw()
#        self.fpsdisplay.draw()
        label = pyglet.text.Label('States per second: {:.2f}'.format(pyglet.clock.get_fps()*self.gamestate.drawevery),
                          font_name='Times New Roman',
                          font_size=12,
                          x=10, y=10,
                          anchor_x='left', anchor_y='bottom')
        label.draw()
#        batch
        

@numba.njit(fastmath=True, parallel=True, debug=False)
def distance(positions, size=(500,500)):
    shape = len(positions)
#    print('a')
    result = np.empty((shape,shape))
    for i in range(shape):
        result[i,i] = 0.0
        for j in range(i,shape):
#            if i<=j: continue
#            result[i,j] = result[j,i] = positions[i,0]
            dx = min(np.abs(size[0] - positions[i,0] - positions[j,0]),
                     np.abs(positions[i,0] - positions[j,0]))
            dy = min(np.abs(size[1] - positions[i,1] - positions[j,1]),
                     np.abs(positions[i,1] - positions[j,1]))
            
            result[i,j] = result[j,i] = (dx**2+dy**2)**(0.5)
    return result

#def distance_one_loop(positions, size=(500,500)):
#    shape = len(positions)
##    print('a')
#    result = np.empty((shape,shape))
#    for i in range(shape):
##        result[i,i] = 0.0
#        p = positions[i,:]
#        
#        result[i,:] = np.sum((positions - p)**2, axis=1)
#    return result
#
#@numba.njit()
#def distance_one_loop2(positions, size=(500,500)):
#    shape = len(positions)
##    print('a')
#    result = np.empty((shape,shape))
#    for i in range(shape):
##        result[i,i] = 0.0        
#        result[i,:i] = result[:i, i] = np.sum((positions[:i,:] - positions[i,:])**2, axis=1)
#    return result

def average_circular_directions(positions, directions, radius, size=(100,100)):
    dfilter = distance(positions, size)<=radius
    return np.angle(dfilter@(np.exp(1j * directions))/(np.sum(dfilter,axis=1)))

def circular_mean(angles):
    return np.angle(np.mean(np.exp(1j*angles)))

def circular_variance(angles):
    return np.sqrt(np.sum(((np.exp(1j*circular_mean(angles))-np.exp(1j*angles))**2)))

def polarisation(angles):
    return np.abs(np.sum(np.exp(1j*angles)))/len(angles)

if __name__ == '__main__':
    pyglet.options['vsync'] = True
    game = gamestate(boids=500, drawevery=1, eta=0.1, radius=50, size=(1000,1000))    
    window = viewport(game)
    window.set_size(1280*2//3,720*2//3)
    
    pyglet.clock.schedule(game.update)
    pyglet.app.run()
    
    pyglet.clock.unschedule(game.update)
    print('done')
    positions = game.positions
    directions = game.directions
    plt.plot(np.array(game.polarisation))
#    plt.plot(game.polarisation)#/np.array(game.polarisation))
    plt.show()
    plt.hist(game.directions/np.pi, int(np.sqrt(len(game.directions))))