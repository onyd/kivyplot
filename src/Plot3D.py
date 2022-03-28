from kivyplot import Scene, Renderer, PerspectiveCamera
from kivy.uix.floatlayout import FloatLayout
from kivyplot import Mesh, Material
import math
from kivyplot.math.transform import vec
from kivyplot.extras.geometries import *

class Plot3D(FloatLayout):

    def __init__(self, radius, *args, **kw):
        super(Plot3D, self).__init__(*args, **kw)
        # Setup plot grid
        grid_geo = GridGeometry(12, 2)
        material = Material(color=(0., 0., 0.), shininess=5)
        self.grid = Mesh(grid_geo, material, mesh_mode='lines',
                         position=vec(-1, 0, -1))

        self.ray_cast = None

        # Setup rendering scene
        self.renderer = Renderer()
        self.renderer.main_light.intensity = 500

        self.scene = Scene()
        self.camera = PerspectiveCamera(15, 1, 1, 1500)
        self.radius = radius

        self.scene.add(self.grid)

        self.points_data = []

        # Use for registering touches for rotation
        self._touches = []

        # Specify the current view mode in {'rotate', 'pick', 'select'} ('select' is not yet implemented)
        self.view_mode = 'rotate'

        # Spherical view angle
        self.theta = 0
        self.phi = 90

        # Setup camera
        self.camera.pos[2] = radius
        self.camera.look_at(np.array([0, 0, 0]))

        self.add_widget(self.renderer)
        self.renderer.bind(size=self._adjust_aspect)

    def render(self):
        self.renderer.render(self.scene, self.camera)

    def reload(self):
        self.renderer.reload(self.scene)

    def add_points(self, *points, color=(0.0, 0.0, 1.0), radius=0.05):
        meshes = []
        geometry = SphereGeometry(radius)
        material = Material(color=color, shininess=10)
        for p in points:
            meshes.append(Mesh(geometry, material, position=p))
        self.points_data.extend([(p, radius) for p in points])
        self.scene.add(*meshes)

    def set_view_mode(self, mode):
        self.view_mode = mode

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.radius -= 5
                    if self.radius < 10:
                        self.radius = 10
                elif touch.button == 'scrollup':
                    self.radius += 5
                self.update_camera()

            if self.view_mode == 'rotate':
                touch.grab(self)
                self._touches.append(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if self.view_mode == 'rotate':
                touch.ungrab(self)
                self._touches.remove(touch)
            # elif self.view_mode == 'pick':
                ray_vec = self.renderer.ray_cast(*touch.pos)
                self.show_ray_cast(ray_vec)

                # Check collision against spheres
                collides = []
                for p, r in self.points_data:
                    v = self.camera.pos - p
                    b = np.vdot(ray_vec, v)
                    c = np.vdot(v, v) - r**2

                    d = b**2-c
                    if d >= 0:
                        collides.append((-b-np.sqrt(d), p))

                #print(len(collides))
                if len(collides) > 0:
                    # Select the nearest sphere to the camera plane
                    nearest = min(collides, key=lambda x: x[0])
                    print(nearest)

    def show_ray_cast(self, ray_vec):
        if self.ray_cast:
            self.scene.remove(self.ray_cast)
        line_geo = LineGeometry(
            vec(0, 0, 0), -5.0 * ray_vec)
        material = Material(color=(1., 0., 0.), shininess=5)
        self.ray_cast = Mesh(line_geo, material, mesh_mode='lines', position=self.camera.pos)
        self.scene.add(self.ray_cast)
        self.reload()

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and self.view_mode == 'rotate':
            if touch in self._touches and touch.grab_current == self:
                if len(self._touches) == 1:
                    self.do_rotate(touch)
                elif len(self._touches) == 2:
                    pass

    def do_rotate(self, touch):
        d_theta = (touch.dx / self.width) * 360
        d_phi = (touch.dy / self.height) * 360

        self.phi += d_phi
        self.theta -= d_theta

        if self.phi > 180:
            self.phi = 180
        elif self.phi <= 0:
            self.phi = 1  # Avoid crash

        self.update_camera()

    def update_camera(self, *args):
        _phi = math.radians(self.phi)
        _theta = math.radians(self.theta)
        z = self.radius * math.cos(_theta) * math.sin(_phi)
        x = self.radius * math.sin(_theta) * math.sin(_phi)
        y = self.radius * math.cos(_phi)
        self.camera.pos = x, y, z
        self.camera.look_at(np.array([0, 0, 0]))

    def _adjust_aspect(self, inst, val):
        rsize = self.renderer.size
        aspect = rsize[0] / float(rsize[1])
        self.renderer.camera.aspect = aspect