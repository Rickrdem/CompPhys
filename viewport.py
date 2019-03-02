import pyglet
import numpy as np
from pyglet.gl.gl import *
from pyglet.gl.glu import *
import pyglet.window.key as key

class Viewport(pyglet.window.Window):
    def __init__(self, gamestate=None, drawevery=1):
        assert gamestate is not None
        super().__init__(resizable=True, vsync=False)

        self.drawevery = drawevery

        self.gamestate = gamestate
        self.fpsdisplay = pyglet.window.FPSDisplay(self)
        
        self.rotation = [1, 0]
        self.fov = 75.
        self.colors = None
        self.batch = pyglet.graphics.Batch()

    def on_draw(self):
        for i in range(self.drawevery-1):
            self.gamestate.update(0)  # calculate self.drawevery-1 extra states before rendering
        self.clear()
        self.set_3d()
        self.particles_to_batch()
        self.draw_bounding_box()
        self.batch.draw()

        self.set_2d()

        self.draw_hud()

    def draw_bounding_box(self):
        verts, faces = cube()
        verts /=2
        # verts /= self.gamestate.size[0]
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        batch = pyglet.graphics.Batch()
        batch.add_indexed(
            len(verts),
            GL_QUADS,
            None,
            faces.flatten(),
            ('v3f', verts.flatten())
        )
        batch.draw()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def particles_to_batch(self):
        gamestate = self.gamestate
        positions = (gamestate.positions - np.asarray(gamestate.size) / 2) / np.asarray(gamestate.size)[np.newaxis, :]
        
        verts, faces = icosahedron()  # The vertices and faces of a single icosahedron

        verts = 1/10* verts / gamestate.size[0]  # Scaling down size of ico to 1 sigma

        verts_per_shape = len(verts)
        faces_per_shape = len(faces)

        particles = gamestate.particles
        dimensions = gamestate.dimensions

        verts = np.tile(verts.T, particles).T  # create vertices for all particles
        verts[:, :min(3, dimensions)] += positions.repeat(verts_per_shape, axis=0)

        faces = (faces.flatten()[:, np.newaxis] + verts_per_shape * (np.arange(particles))).T
        
        if self.colors is None:
            self.colors = (verts.T/np.sum(verts, axis=1)).T

        if hasattr(self, 'spheres'):
            self.spheres.vertices = verts.flatten()
        else:
            self.spheres = self.batch.add_indexed(
                len(verts),
                GL_TRIANGLES,
                None,
                faces.flatten(),
                ('v3f', verts.flatten()),
                ('c3f', self.colors.flatten())
            )

    def draw_hud(self):
        text = """
            Time step size {h}
            Particles {particles}
            Boxsize {boxsize}
            Average velocity {vel:.2f}
            Temperature {temp:.2f}
            Pressure {pressure:.2f}
            Draw every {drawevery}
            States per frame {fps:.2f}
            """.format(h=self.gamestate.h,
                       particles=self.gamestate.particles,
                       boxsize=self.gamestate.size,
                       vel=1,
                       temp=1,
                       pressure=1,
                       drawevery=self.drawevery,
                       fps=pyglet.clock.get_fps() * self.drawevery)

        document = pyglet.text.decode_text(text)
        document.set_style(0, 0, dict(font_name='Arial', font_size=8))

        width, height = self.get_size()

        meta_information = pyglet.text.layout.TextLayout(document, width//2, height, multiline=True)
        meta_information.x = -30
        meta_information.anchor_x = 'left'
        meta_information.y = 0
        meta_information.draw()

        text = """
            Rotate the view by clicking and dragging
            Scrolling zooms in and out of the view
            Horizontal arrow keys alters states per frame
            """
        document = pyglet.text.decode_text(text)
        document.set_style(0, 0, dict(font_name='Arial', font_size=8))

        instructions = pyglet.text.layout.TextLayout(document, width, height, multiline=False)
        instructions.x = -30
        instructions.y = -height+15
        instructions.anchor_x = 'left'
        instructions.anchor_y = 'bottom'

        instructions.draw()


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # if buttons & self.mouse.LEFT:
        x, y = self.rotation
        x, y = x + dx / 5, y + dy / 5
        self.rotation = (x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol==key.RIGHT:
            self.drawevery *= 2
            if self.drawevery >= 1000:
                self.drawevery = 1000
        elif symbol==key.LEFT:
            self.drawevery = self.drawevery//2
            if self.drawevery <=1:
                self.drawevery = 1
        elif symbol==key.V:
            self.set_vsync(not self.vsync)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.fov -= scroll_y
        if self.fov <= 1:
            self.fov =1
        elif self.fov >= 180:
            self.fov = 180

    def set_2d(self):
        """ Configure OpenGL for drawing 2d objects such as the fps-counter
        """
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)

        # Ortho projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        """ Configure OpenGL for drawing 3d objects
        """
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)

        # set perspective transformation to unity

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, width / float(height), 0.1, 3)

        glTranslatef(0, 0, -1.1)

        # set modelview rotation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        theta, phi = self.rotation
        glRotatef(theta, 0, 1, 0)
        glRotatef(-phi, np.cos(np.deg2rad(theta)), 0, np.sin(np.deg2rad(phi)))

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


def setup():
    """ Basic OpenGL configuration.
    """
    # Set the color of "clear", i.e. the sky, in rgba.

    glClearColor(0.5, 0.69, 1.0, 1)

    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    glEnable(GL_CULL_FACE)


def triangle():
    vertices = np.asarray([
        [-1, -.5, 0],
        [1, -0.5, 0],
        [0, 1, 0]
    ])
    faces = np.asarray([
        [0, 1, 2]
    ])
    return vertices, faces


def cube():
    vertices = np.array([
        [-1., -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],

        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])
    faces = np.array([
        [0, 1, 2, 3],
        [3, 2, 6, 7],
        [7, 6, 5, 4],
        [6, 2, 1, 5],
        [3, 7, 4, 0],
        [4, 5, 1, 0]
    ])
    return vertices, faces


def icosahedron():
    t = 1 + np.sqrt(5) / 2
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
    vertices /= (np.sqrt(t**2+1))  # unit size

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
