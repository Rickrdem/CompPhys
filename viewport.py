import pyglet
import pyglet.gl.glu as glu
import pyglet.gl.gl as gl
import numpy as np
from itertools import cycle


class Viewport(pyglet.window.Window):
    def __init__(self, gamestate=None):
        assert gamestate is not None
        super().__init__()
        self.gamestate = gamestate
        self.fpsdisplay = pyglet.window.FPSDisplay(self)

        self.rotation = [1, 0]
        self.colors = None

    def on_draw(self):
        self.clear()
        self.batch = pyglet.graphics.Batch()
        self.set_3d()
        self.particles_to_batch()
        # self.bounding_box_to_batch()
        self.batch.draw()

        self.set_2d()

        self.draw_fps()

    def draw_bounding_box(self):
        verts, faces = cube()
        verts /=2
        # verts /= self.gamestate.size[0]
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.batch.add_indexed(
            len(verts),
            gl.GL_TRIANGLES,
            None,
            faces.flatten(),
            ('v3f', verts.flatten())
        )
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def particles_to_batch(self):
        gamestate = self.gamestate
        # print(gamestate.positions.shape, gamestate.size)

        # positions = gamestate.positions - np.asarray(gamestate.size)/2
        positions = (gamestate.positions - np.asarray(gamestate.size) / 2) / np.asarray(gamestate.size)[np.newaxis,:]
        verts, faces = icosahedron()  # The vertices and faces of a single icosahedron
        print(verts)
        verts = verts/ 100  # Scaling down size of ico. Should probably be scaled down to ~sigma
        verts_per_shape = len(verts)

        particles = gamestate.particles
        dimensions = gamestate.dimensions

        verts = np.tile(verts.T, particles).T  # create vertices for all particles
        verts[:, :min(3, dimensions)] += positions.repeat(verts_per_shape).reshape(dimensions, -1).T

        faces = (faces.flatten()[:, np.newaxis] + verts_per_shape * np.arange(particles)).T.flatten()
        faces = faces.flatten()

        if self.colors is None:
            self.colors = (verts.T/np.sum(verts, axis=1)).T
        spheres = self.batch.add_indexed(
            len(verts),
            gl.GL_TRIANGLES,
            None,
            faces,
            ('v3f', verts.flatten()),
            ('c3f', self.colors.flatten())
        )

    def draw_fps(self):
        label = pyglet.text.Label('States per second: {:.2f}'.format(pyglet.clock.get_fps() * self.gamestate.drawevery),
                                  font_name='Times New Roman',
                                  font_size=12,
                                  x=10, y=15,
                                  anchor_x='left', anchor_y='top',
                                  color=(0, 0, 0, 255))
        label.draw()


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # if buttons & self.mouse.LEFT:
        x, y = self.rotation
        x, y = x + dx / 5, y + dy / 5
        self.rotation = (x, y)

    def set_2d(self):
        """ Configure OpenGL for drawing 2d objects such as the fps-counter
        """
        width, height = self.get_size()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)

        # Ortho projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def set_3d(self):
        """ Configure OpenGL for drawing 3d objects
        """
        width, height = self.get_size()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)

        # set perspective transformation to unity
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(70.0, width / float(height), 1, 50)

        gl.glTranslatef(0,0,-2)

        # set modelview rotation
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        theta, phi = self.rotation
        gl.glRotatef(theta, 0, 1, 0)
        gl.glRotatef(-phi, np.cos(np.deg2rad(theta)), 0, np.sin(np.deg2rad(phi)))


        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


def setup():
    """ Basic OpenGL configuration.
    """
    # Set the color of "clear", i.e. the sky, in rgba.
    gl.glClearColor(0.5, 0.69, 1.0, 1)
    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    gl.glEnable(gl.GL_CULL_FACE)

def triangle():
    vertices = np.asarray([
        [-1,-.5,0],
        [1,-0.5,0],
        [0,1,0]
    ])
    faces = np.asarray([
        [0, 1, 2]
    ])
    return vertices, faces

def cube():
    vertices = np.array([
        [-1.,-1,-1],
        [1,-1,-1],
        [1,1,-1],
        [-1,1,-1],

        [-1,-1,1],
        [1,-1,1],
        [1,1,1],
        [-1,1,1]
    ])

    faces = np.array([
        [0,1,2],
        [2,3,0],

        [3,2,6],
        [6,7,3],

        [7,6,5],
        [5,4,7],

        [3,7,4],
        [4,0,3],

        [6,2,1],
        [1,5,6],

        [4,5,1],
        [1,0,4]
    ])

    return vertices, faces

def icosahedron():
    t = 1 + np.sqrt(5) / 2
    # vertices = list(cycle((1, t, 0))) +\
    #     list(cycle((-1, t, 0))) +\
    #     list(cycle((1, -t, 0))) +\
    #     list(cycle((-1, -t, 0)))
    vertices = np.asarray([
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],

        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],

        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1]
    ])

    faces = np.asarray([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])
    return vertices, faces
