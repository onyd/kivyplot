
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


import numpy as np
import os
import kivyplot


from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.fbo import Fbo
from kivy.core.window import Window
from kivy.graphics import BindTexture
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics.opengl import glEnable, glDisable, glCullFace, GL_DEPTH_TEST, GL_CULL_FACE, GL_BACK
from kivy.graphics.transformation import Matrix
from kivy.graphics import (
    Callback, PushMatrix, PopMatrix,
    Rectangle, Canvas
)
import time

kivyplot_path = os.path.abspath(os.path.dirname(kivyplot.__file__))


class RendererError(Exception):
    pass


class Renderer(Widget):

    def __init__(self, picking=False, **kw):
        self.shader_file = kw.pop("shader_file", None)
        self.picking_shader_file = kw.pop("picking_shader_file", None)
        self.picking = picking
        self.canvas = Canvas()
        super(Renderer, self).__init__(**kw)

        with self.canvas:
            self._viewport = Rectangle(size=self.size, pos=self.pos)
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=True, compute_normal_mat=True,
                           clear_color=(1., 1., 1., 1.))
            if self.picking:
                self.picking_fbo = Fbo(size=self.size,
                                       with_depthbuffer=True, compute_normal_mat=False,
                                       clear_color=(0., 0., 0., 1.))

        self._config_fbo()
        if self.picking:
            self._config_picking_fbo()
        self.texture = self.fbo.texture
        self.camera = None
        self.scene = None

    def _config_fbo(self):
        # set shader file here
        self.fbo.shader.source = self.shader_file or \
            os.path.join(kivyplot_path, "default.glsl")
        with self.fbo:
            PushMatrix()
            Callback(self._setup_gl_context)
            # instructions set for all instructions
            self._instructions = InstructionGroup()
            Callback(self._reset_gl_context)
            PopMatrix()

    def _config_picking_fbo(self):
        # set shader file here
        self.picking_fbo.shader.source = self.picking_shader_file or \
            os.path.join(kivyplot_path, "picking_default.glsl")

        with self.picking_fbo:
            PushMatrix()
            Callback(self._setup_gl_context_picking)
            # instructions set for all instructions
            self._instructions_picking = InstructionGroup()
            Callback(self._reset_gl_context)
            PopMatrix()

    def _setup_gl_context(self, *args):
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        self.fbo.clear_buffer()

    def _reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

    def _setup_gl_context_picking(self, *args):
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        self.picking_fbo.clear_buffer()

    def reload(self):
        self._instructions.clear()
        self._instructions.add(self.scene.as_instructions())

        if self.picking:
            self._instructions_picking.clear()
            self._instructions_picking.add(self.scene.as_picking_instructions())

    def render(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.camera.bind_to(self)
        
        self._instructions.add(scene.as_instructions())
        if self.picking:
            self._instructions_picking.add(scene.as_picking_instructions())

        Clock.schedule_once(self._update_matrices, -1)

    def on_size(self, instance, value):
        if value[0] > 0 and value[1] > 0:
            self.fbo.size = value
            if self.picking:
                self.picking_fbo.size = value

            self._viewport.texture = self.fbo.texture
            self._viewport.size = value
            self._viewport.pos = self.pos
            self._update_matrices()

    def on_pos(self, instance, value):
        self._viewport.pos = self.pos
        self._update_matrices()

    def on_texture(self, instance, value):
        self._viewport.texture = value

    def _update_matrices(self, dt=None):
        if self.camera:
            self.fbo['projection_mat'] = self.camera.projection_matrix
            self.fbo['modelview_mat'] = self.camera.modelview_matrix
            self.fbo['model_mat'] = self.camera.model_matrix
            self.fbo['camera_pos'] = [float(p) for p in self.camera.position]
            self.fbo['camera_back'] = [float(p) for p in self.camera._back]
            self.fbo['t'] = time.time()
            self.fbo['view_mat'] = Matrix().rotate(
                Window.rotation, 0.0, 0.0, 1.0)

            if self.picking:
                self.picking_fbo['projection_mat'] = self.camera.projection_matrix
                self.picking_fbo['modelview_mat'] = self.camera.modelview_matrix
                self.picking_fbo['model_mat'] = self.camera.model_matrix
                self.picking_fbo['camera_pos'] = [
                    float(p) for p in self.camera.position]
                self.picking_fbo['view_mat'] = Matrix().rotate(
                    Window.rotation, 0.0, 0.0, 1.0)
        else:
            raise RendererError("Camera is not defined for renderer")

    def set_clear_color(self, color):
        self.fbo.clear_color = color

    def get_cid_at(self, x, y):
        if self.picking:
            cid = tuple(self.picking_fbo.get_pixel_color(x, y))
            if cid != (0, 0, 0, 255):
                return cid[:3]
            return None
        else:
            raise ValueError("picking must be True to use picking feature")

    # def ray_cast(self, x, y):
    #     # Convert mouse pos to homogeneous normalized coordinates
    #     x = 2 * x / self._viewport.size[0] - 1
    #     y = 2 * y / self._viewport.size[1] - 1
    #     z = -1.0
    #     w = 1.0

    #     # Inverse projection
    #     ray_eye = self.camera.projection_matrix.inverse().tolist() @ np.array([x, y, z, w])
    #     ray_eye[2] = -1.0
    #     ray_eye[3] = 0.0

    #     # Inverse view
    #     ray_world = self.camera.modelview_matrix.inverse().tolist() @ ray_eye[:3]

    #     ray_world[0] *= -1
    #     ray_world[1] *= -1

    #     return ray_world / np.linalg.norm(ray_world)
