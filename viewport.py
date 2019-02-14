import pyglet
import numpy as np

class Viewport(pyglet.window.Window):
    def __init__(self, gamestate=None):
        assert gamestate != None
        super().__init__()
        self.gamestate = gamestate

        self.fpsdisplay = pyglet.window.FPSDisplay(self)

    def on_draw(self):
        self.clear()

        gamestate = self.gamestate
        x, y = gamestate.positions.T
        linelength = 10
        windowsize = self.get_size()

        x = windowsize[0] * x / gamestate.size[0]
        y = windowsize[1] * y / gamestate.size[1]

        dx, dy = np.cos(gamestate.directions) * linelength, np.sin(gamestate.directions) * linelength

        batch = pyglet.graphics.Batch()

        lines = np.dstack((x, y, x + dx, y + dy)).flatten()
        batch.add(2 * len(x), pyglet.gl.GL_LINES, None,
                  ('v2f',
                   lines))

        batch.draw()
        label = pyglet.text.Label('States per second: {:.2f}'.format(pyglet.clock.get_fps() * self.gamestate.drawevery),
                                  font_name='Times New Roman',
                                  font_size=12,
                                  x=10, y=10,
                                  anchor_x='left', anchor_y='bottom')
        label.draw()
#        batch