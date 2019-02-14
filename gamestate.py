import numpy as np

class Gamestate():
    def randomise_state(self):
        self.positions = np.random.randn(self.boids, 2)
        self.positions[:, 0] *= self.size[0]
        self.positions[:, 1] *= self.size[1]
        self.directions = np.random.uniform(0, 2 * np.pi, self.boids)  # np.random.rand(self.boids)*np.pi%(2*np.pi)
        print(np.sum(np.exp(1j * self.directions)))
        print(circular_mean(self.directions) / np.pi)

    #        print("boids",len(self.directions))
    #        plt.hist(self.directions, bins=int(np.sqrt(len(self.directions))))

    def __init__(self, boids=2000, size=(500, 500), eta=0.3, radius=10,
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
            noise = np.random.uniform(-np.pi, np.pi, self.boids)

            self.directions = average_circular_directions(self.positions,
                                                          self.directions,
                                                          self.radius)  # *(1-self.eta)
            self.directions += noise * self.eta

            self.positions += np.array([np.cos(self.directions),
                                        np.sin(self.directions)]).T
            self.positions[:, 0] %= self.size[0]
            self.positions[:, 1] %= self.size[1]

            self.mean.append(circular_mean(self.directions))
            self.polarisation.append(polarisation(self.directions))
            self.directions %= (2 * np.pi)