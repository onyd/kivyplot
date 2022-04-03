"""
The MIT License (MIT)

Copyright (c) 2013 Niko Skrypnik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


__all__ = ('Camera', )

from kivy.properties import NumericProperty, ObjectProperty
from kivy.graphics.transformation import Matrix
import numpy as np
from ..core.object3d import Object3D

class Camera(Object3D):
    """
    Base camera class
    """

    scale = NumericProperty(1.0)
    _right = ObjectProperty(np.array([1, 0, 0]), force_dispatch=True)
    _up = ObjectProperty(np.array([0, 1, 0]), force_dispatch=True)
    _back = ObjectProperty(np.array([0, 0, 1]), force_dispatch=True)
    up = ObjectProperty(np.array([0, 1, 0]), force_dispatch=True)

    def __init__(self):
        super(Camera, self).__init__()
        self.projection_matrix = Matrix()
        self.modelview_matrix = Matrix()
        self.model_matrix = Matrix()
        self.viewport_matrix = (0, 0, 0, 0)
        self.renderer = None  # renderer camera is bound to
        self._look_at = None
        self.look_at(np.array([0, 0, -1]))

    def on_pos_changed(self, coord, v):
        """ Camera position was changed """
        self.look_at(self._look_at)

    def on_up(self, instance, up):
        """ Camera up vector was changed """
        pass

    def on_scale(self, instance, scale):
        """ Handler for change scale parameter event """

    def look_at(self, *v):
        if len(v) == 1:
            v = v[0]

        m = Matrix()
        m = m.look_at(
            self.pos[0], self.pos[1], self.pos[2],
            v[0], v[1], v[2],
            self.up[0], self.up[1], self.up[2])
        self.modelview_matrix = m
        # # set camera vectors from view matrix
        self._right = np.array([m[0], m[1], m[2]])
        self._up = np.array([m[4], m[5], m[6]])
        self._back = np.array([m[8], m[9], m[10]])
        self._look_at = v
        self.update()

    def bind_to(self, renderer):
        """ Bind this camera to renderer """
        self.renderer = renderer

    def update(self):
        if self.renderer:
            self.viewport_matrix = (
                self.renderer._viewport.pos[0],
                self.renderer._viewport.pos[1],
                self.renderer._viewport.size[0],
                self.renderer._viewport.size[1]
            )
            self.model_matrix = self.modelview_matrix.multiply(
                self.renderer.fbo['view_mat'].inverse())
            self.renderer._update_matrices()

    def update_projection_matrix(self):
        """ This function should be overridden in the subclasses
        """
        raise NotImplementedError()
