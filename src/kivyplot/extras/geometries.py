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
from kivyplot.core.geometry import Geometry
from kivyplot.core.face3 import Face3
from kivyplot import VertexGroup


class BoxGeometry(Geometry):

    _cube_vertices = [(-1, 1, -1), (1, 1, -1),
                      (1, -1, -1), (-1, -1, -1),
                      (-1, 1, 1), (1, 1, 1),
                      (1, -1, 1), (-1, -1, 1),
                      ]

    _cube_faces = [(0, 1, 2), (0, 2, 3), (3, 2, 6),
                   (3, 6, 7), (7, 6, 5), (7, 5, 4),
                   (4, 5, 1), (4, 1, 0), (4, 0, 3),
                   (7, 4, 3), (5, 1, 2), (6, 5, 2)
                   ]

    _cube_normals = [(0, 0, 1), (-1, 0, 0), (0, 0, -1),
                     (1, 0, 0), (0, 1, 0), (0, -1, 0)
                     ]

    def __init__(self, width, height, depth, **kw):
        name = kw.pop('name', '')
        super(BoxGeometry, self).__init__(name)
        self.width_segment = kw.pop('width_segment', 1)
        self.height_segment = kw.pop('height_segment', 1)
        self.depth_segment = kw.pop('depth_segment', 1)

        self.w = width
        self.h = height
        self.d = depth

        self._build_box()

    def _build_box(self):
        for v in self._cube_vertices:
            v = np.array([0.5 * v[0] * self.w,
                          0.5 * v[1] * self.h,
                          0.5 * v[2] * self.d])
            self.vertices.append(v)

        n_idx = 0
        for f in self._cube_faces:
            normal = self._cube_normals[int(n_idx / 2)]
            face3 = Face3(*f, normals=[normal, normal, normal])
            n_idx += 1
            self.vertex_groups.append(face3)


class SphereGeometry(Geometry):
    def __init__(self, radius, stacks=12, sectors=24, **kw):
        name = kw.pop('name', '')
        super(SphereGeometry, self).__init__(name)
        self.radius = radius
        self.stacks = stacks
        self.sectors = sectors

        self._build_sphere()

    def _build_sphere(self):
        lengthInv = 1.0 / self.radius

        # Build verticies and normals
        sector_step = 2 * np.pi / self.sectors
        stack_step = np.pi / self.stacks
        normals = []
        for i in range(self.stacks + 1):
            stack_angle = np.pi / 2 - i * stack_step
            xy = self.radius * np.cos(stack_angle)
            z = self.radius * np.sin(stack_angle)

            for j in range(self.sectors + 1):

                sector_angle = j * sector_step

                x = xy * np.cos(sector_angle)
                y = xy * np.sin(sector_angle)
                self.vertices.append(np.array([x, y, z]))

                nx = x * lengthInv
                ny = y * lengthInv
                nz = z * lengthInv
                normals.append(np.array([nx, ny, nz]))

        # Build faces of the sphere
        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1

            for j in range(self.sectors):

                if i != 0:
                    face = Face3(
                        k1, k2, k1 + 1, normals=[normals[k1], normals[k2], normals[k1+1]])
                    self.vertex_groups.append(face)

                if i != self.stacks-1:
                    face = Face3(
                        k1 + 1, k2, k2 + 1, normals=[normals[k1+1], normals[k2], normals[k2+1]])
                    self.vertex_groups.append(face)

                k1 += 1
                k2 += 1


class GridGeometry(Geometry):
    def __init__(self, slices, size, **kw):
        name = kw.pop('name', '')
        super(GridGeometry, self).__init__(name)
        self.width_segment = kw.pop('width_segment', 1)
        self.height_segment = kw.pop('height_segment', 1)

        self.slices = slices
        self.size = size

        self._build_grid()

    def _get_index(self, i, j):
        return i * (self.slices + 1) + j

    def _build_grid(self):
        k = 0
        for i in range(1, self.slices):
            self.vertices.append(np.array([i * self.size / self.slices, 0, 0]))
            self.vertices.append(
                np.array([i * self.size / self.slices, 0, self.size]))
            self.vertex_groups.append(VertexGroup(k, k+1))
            k += 2

            self.vertices.append(np.array([0, 0, i * self.size / self.slices]))
            self.vertices.append(
                np.array([self.size, 0, i * self.size / self.slices]))
            self.vertex_groups.append(VertexGroup(k, k+1))
            k += 2

        self.vertices.append(np.array([0, 0, 0]))
        self.vertices.append(np.array([0, 0, self.size]))
        self.vertex_groups.append(VertexGroup(k, k+1))
        k += 2

        self.vertices.append(np.array([0, 0, self.size]))
        self.vertices.append(np.array([self.size, 0, self.size]))
        self.vertex_groups.append(VertexGroup(k, k+1))
        k += 2

        self.vertices.append(np.array([self.size, 0, self.size]))
        self.vertices.append(np.array([self.size, 0, 0]))
        self.vertex_groups.append(VertexGroup(k, k+1))
        k += 2

        self.vertices.append(np.array([self.size, 0, 0]))
        self.vertices.append(np.array([0, 0, 0]))
        self.vertex_groups.append(VertexGroup(k, k+1))
        k += 2


class LineGeometry(Geometry):
    def __init__(self, p1, p2, **kw):
        name = kw.pop('name', '')
        super(LineGeometry, self).__init__(name)
        self.width_segment = kw.pop('width_segment', 1)
        self.height_segment = kw.pop('height_segment', 1)

        self.p1 = p1
        self.p2 = p2

        self._build_grid()

    def _build_grid(self):
        self.vertices.extend([self.p1, self.p2])
        self.vertex_groups.append(VertexGroup(0, 1))
