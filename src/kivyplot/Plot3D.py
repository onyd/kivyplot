from kivyplot import Scene, Renderer, PerspectiveCamera
from kivy.uix.floatlayout import FloatLayout
from kivyplot import Mesh, Material
import math
from kivyplot.extras.geometries import *
from kivy.properties import NumericProperty

class Plot3D(FloatLayout):
    radius = NumericProperty(20)

    def __init__(self, **kw):
        super(Plot3D, self).__init__(**kw)

        #self.ray_cast = None
        self.cid_to_mesh = {}

        # Setup rendering scene
        self.renderer = Renderer(picking=False)

        self.scene = Scene()
        self.camera = PerspectiveCamera(20, 1, 0.1, 500)

        self.points_geo = {}
        self.points_mat = {}
        self.setup_grid()

        # Use for registering touches for rotation
        self._touches = []

        # Specify the current view mode in {'orbit', 'pick', 'select'} ('pick' and 'select' is not yet implemented)
        self.view_mode = 'orbit'

        # Spherical view angle
        self.theta = 0
        self.phi = 90

        # Setup camera
        self.camera.pos[2] = self.radius
        self.camera.look_at(np.array([0, 0, 0]))

        self.add_widget(self.renderer)

    def setup_grid(self):
        grid_geo = GridGeometry(12, 2)
        material = Material(color=(0., 0., 0.), shininess=5)
        self.grid = Mesh(grid_geo, material, mesh_mode='lines',
                         position=np.array([-1, 0, -1]))
        self.scene.add(self.grid, group="__grid__")

    def render(self):
        self.renderer.bind(size=self._adjust_aspect)
        self.renderer.render(self.scene, self.camera)

    def reload(self):
        self.renderer.reload()

    def show_group(self, group):
        self.scene.show_group(group)

    def hide_group(self, group):
        self.scene.hide_group(group)

    def add_group(self, group, color=(0.0, 0.0, 1.0), radius=0.05):
        assert(group != "__grid__")
        self.points_mat[group] = Material(
            color=color, shininess=10)
        self.points_geo[group] = SphereGeometry(radius)

    def add_points(self, *points, group=None, color=(0, 0, 1), radius=0.05):
        assert(group != "__grid__")
        if group is None:
            geometry = SphereGeometry(radius)
            material = Material(color=color, shininess=10)
        else:
            geometry = self.points_geo[group]
            material = self.points_mat[group]

        # Build meshes
        meshes = []
        for p in points:
            m = Mesh(geometry, material, position=p,
                     data={"pos": p, "radius": radius})
            meshes.append(m)
            self.cid_to_mesh[m.get_cid()] = m
        self.scene.add(*meshes, group=group)

    def remove_groups(self, *groups):
        assert("__grid__" not in groups)

        for group in groups:
            self.scene.clear(group)
            del self.points_mat[group]
            del self.points_geo[group]

    def clear(self):
        for group in list(self.points_geo.keys()):
            self.remove_groups(group)
        self.scene.clear()

    def set_mode(self, mode):
        self.view_mode = mode

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.view_mode == 'orbit':
                if touch.is_mouse_scrolling:
                    if touch.button == 'scrolldown':
                        self.radius -= 5
                        if self.radius < 10:
                            self.radius = 10
                    elif touch.button == 'scrollup':
                        self.radius += 5
                    self.update_camera()
                    return True

                touch.grab(self)
                self._touches.append(touch)
                return True

        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if self.view_mode == 'orbit':
                if self._touches:
                    touch.ungrab(self)
                    self._touches.remove(touch)
                    return True
            elif self.view_mode == 'pick':
                cid = self.renderer.get_cid_at(*touch.pos)
                if cid:
                    print(self.cid_to_mesh[cid])

        return super().on_touch_up(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and self.view_mode == 'orbit':
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
        self.camera.pos = np.array([x, y, z])

    def _adjust_aspect(self, inst, val):
        try:
            rsize = self.renderer.size
            aspect = rsize[0] / float(rsize[1])
            self.renderer.camera.aspect = aspect
        except:
            pass
