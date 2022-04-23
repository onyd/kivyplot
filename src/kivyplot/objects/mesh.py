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

from kivy.graphics import Mesh as KivyMesh
import numpy as np
from ..core.object3d import Object3D
from ..core.materials import Material, Selection

DEFAULT_VERTEX_FORMAT = [
    (b'v_pos', 3, 'float'),
    (b'v_normal', 3, 'float')
]
DEFAULT_MESH_MODE = 'triangles'
id = 0


class Mesh(Object3D):

    def __init__(self, geometry, material, mesh_mode=DEFAULT_MESH_MODE, position=np.array([0, 0, 0]), data={}, **kw):
        super(Mesh, self).__init__(position=position, **kw)
        global id
        id += 1
        self.id = id

        self.geometry = geometry
        self.material = material
        self.mtl = self.material  # shortcut for material property
        self.sel_state = Selection()
        self.vertex_format = kw.pop('vertex_format', DEFAULT_VERTEX_FORMAT)
        self.mesh_mode = mesh_mode
        self.data = data
        self.create_mesh()

    def get_cid(self):
        """Convert the id to a color tuple used for picking"""
        r = self.id & 255
        g = (self.id & (255 << 8)) >> 8
        b = (self.id & (255 << 16)) >> 16
        return r, g, b

    def create_mesh(self):
        """ Create real mesh object from the geometry and material """
        vertices = []
        indices = []
        idx = 0
        for vertex_group in self.geometry.vertex_groups:
            for i, v_idx in enumerate(vertex_group.indicies):
                vertex = self.geometry.vertices[v_idx]
                vertices.extend(vertex)

                # Add normals if they are given
                try:
                    normal = vertex_group.vertex_normals[i]
                except:
                    normal = np.array([0, 0, 0])
                vertices.extend(normal)

                indices.append(idx)
                idx += 1
        if idx >= 65535 - 1:
            msg = 'Mesh must not contain more than 65535 indices, {} given'
            raise ValueError(msg.format(idx + 1))
        kw = dict(
            vertices=vertices,
            indices=indices,
            fmt=self.vertex_format,
            mode=self.mesh_mode
        )
        self._mesh = KivyMesh(**kw)
        self._picking_mesh = KivyMesh(**kw)

    def custom_instructions(self, picking):
        if picking:
            yield Material(color=tuple((c / 255.0 for c in self.get_cid())), transparency=1)
            yield self._picking_mesh
        else:
            yield self.material
            yield self.sel_state
            yield self._mesh
