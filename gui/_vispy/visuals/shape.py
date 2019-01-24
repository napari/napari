# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


"""
Simple shapes visual based on MeshVisual, LineVisual, and MarkersVisual
"""

from __future__ import division

import numpy as np

from vispy.visuals.visual import CompoundVisual
from vispy.visuals.mesh import MeshVisual
from vispy.visuals.line import LineVisual
from vispy.visuals.markers import MarkersVisual
from vispy.color import Color
from vispy.gloo import set_state


class ShapeVisual(CompoundVisual):
    """
    Displays 2D shapes
    Parameters
    ----------
    pos : array
        Set of vertices defining the polygon.
    color : str | tuple | list of colors
        Fill color of the polygon.
    border_color : str | tuple | list of colors
        Border color of the polygon.
    border_width : int
        Border width in pixels.
        Line widths > 1px are only
        guaranteed to work when using `border_method='agg'` method.
    vertex_size : int
        Vertex size in pixels.
    border_method : str
        Mode to use for drawing the border line (see `LineVisual`).
            * "agg" uses anti-grain geometry to draw nicely antialiased lines
              with proper joins and endcaps.
            * "gl" uses OpenGL's built-in line rendering. This is much faster,
              but produces much lower-quality results and is not guaranteed to
              obey the requested line width or join/endcap styles.
    triangulate : boolean
        Triangulate the set of vertices
    **kwargs : dict
        Keyword arguments to pass to `CompoundVisual`.
    """
    def __init__(self, mesh_vertices=None, mesh_faces=None, lines_vertices=None,
                 lines_connect=None, marker_vertices=None, edge_width=1,
                 edge_color='black', face_color='white', marker_symbol='o',
                 marker_size=10):

        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._lines_vertices = lines_vertices
        self._lines_connect = lines_connect
        self._marker_vertices = marker_vertices

        self._mesh = MeshVisual()
        self._line = LineVisual()
        self._markers = MarkersVisual()

        self._line_highlight = LineVisual()
        self._markers_highlight = MarkersVisual()

        self._edge_color = Color(edge_color)
        self._face_color = Color(face_color)
        self._marker_symbol = marker_symbol
        self._marker_size = marker_size
        self._edge_width = edge_width

        self._update()
        #self._update_highlight()
        CompoundVisual.__init__(self, [self._markers, self._line, self._mesh,
                                self._line_highlight, self._markers_highlight])

        # self.mesh.set_gl_state(polygon_offset_fill=True,
        #                        polygon_offset=(1, 1), cull_face=False)
        self.freeze()

    def _update(self):
        if not self.face_color.is_blank:
            self.mesh.set_data(vertices=self.mesh_vertices,
                               faces=self.mesh_faces,
                               color=self.face_color.rgba)
        else:
            self.mesh.set_data(vertices=None)

        if not self.edge_color.is_blank:
            self.line.set_data(pos=self.lines_vertices,
                               connect=self.lines_connect,
                               color=self.edge_color.rgba,
                               width=self.edge_width)
        else:
            self.line.set_data(width=0)
#        self.line.update()

        if self.marker_vertices is None:
            self.markers.set_data(pos=np.empty((0, 2)))
        else:
            if not self.face_color.is_blank and not self.edge_color.is_blank:
                self.markers.set_data(pos=self.marker_vertices,
                                      size=self.marker_size,
                                      face_color=self.face_color.rgba,
                                      edge_width=self.edge_width, scaling=True,
                                      symbol=self.marker_symbol,
                                      edge_color=self.edge_color.rgba)
            elif not self.face_color.is_blank:
                self.markers.set_data(pos=self.marker_vertices,
                                      size=self.marker_size,
                                      face_color=self.face_color.rgba,
                                      edge_width=self.edge_width, scaling=True,
                                      symbol=self.marker_symbol)
            elif not self.edge_color.is_blank:
                self.markers.set_data(pos=self.marker_vertices,
                                      size=self.marker_size,
                                      edge_color=self.edge_color.rgba,
                                      edge_width=self.edge_width, scaling=True,
                                      symbol=self.marker_symbol)
            else:
                self.markers.set_data(pos=np.empty((0, 2)))
#        self.markers.update()

    @property
    def mesh_vertices(self):
        """ The vertex positions of triangles in the meshes.
        """
        return self._mesh_vertices

    @mesh_vertices.setter
    def mesh_vertices(self, mesh_vertices):
        self._mesh_vertices = mesh_vertices
        self._update()

    @property
    def mesh_faces(self):
        """ The vertex indices of triangles in the meshes.
        """
        return self._mesh_faces

    @mesh_faces.setter
    def mesh_faces(self, mesh_faces):
        self._mesh_faces = mesh_faces
        self._update()

    @property
    def lines_vertices(self):
        """ The vertex positions of lines.
        """
        return self._lines_vertices

    @lines_vertices.setter
    def lines_vertices(self, lines_vertices):
        self._lines_vertices = lines_vertices
        self._update()

    @property
    def lines_connect(self):
        """ The vertex indices of the lines to connect.
        """
        return self._lines_connect

    @lines_connect.setter
    def lines_connect(self, lines_connect):
        self._lines_connect = lines_connect
        self._update()

    @property
    def marker_vertices(self):
        """ The vertex positions of the markers.
        """
        return self._marker_vertices

    @marker_vertices.setter
    def marker_vertices(self, marker_vertices):
        self._marker_vertices = marker_vertices
        self._update()

    @property
    def edge_color(self):
        """ The color of the lines.
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = Color(edge_color)
        self._update()

    @property
    def face_color(self):
        """ The color of the faces.
        """
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = Color(face_color)
        self._update()

    @property
    def marker_symbol(self):
        """ The symbol of the markers.
        """
        return self._marker_symbol

    @marker_symbol.setter
    def marker_symbol(self, marker_symbol):
        self._marker_symbol = marker_symbol
        self._update()

    @property
    def marker_size(self):
        """ The size of the markers.
        """
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size):
        self._marker_size = marker_size
        self._update()

    @property
    def edge_width(self):
        """ The width of all the lines.
        """
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width
        self._update()

    @property
    def mesh(self):
        """The vispy.visuals.MeshVisual that is owned by the ShapeVisual.
           It is used to fill in the polygon
        """
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        self._update()

    @property
    def line(self):
        """The vispy.visuals.LineVisual that is owned by the ShapeVisual.
           It is used to draw all the lines and edges
        """
        return self._line

    @line.setter
    def line(self, line):
        self._line = line
        self._update()

    @property
    def markers(self):
        """The vispy.visuals.MarkersVisual that is owned by the ShapeVisual.
           It is used to draw the vertices of the polygon
        """
        return self._markers

    @markers.setter
    def markers(self, markers):
        self._markers = markers
        self._update()

    def set_data(self, mesh_vertices=None, mesh_faces=None, lines_vertices=None,
                 lines_connect=None, marker_vertices=None, edge_width=1,
                 edge_color='black', face_color='white', marker_symbol='o',
                 marker_size=10):
        """Set the data used to draw this visual.
            Parameters
            ----------
        """
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._lines_vertices = lines_vertices
        self._lines_connect = lines_connect
        self._marker_vertices = marker_vertices
        self._edge_color = Color(edge_color)
        self._face_color = Color(face_color)
        self._marker_symbol = marker_symbol
        self._marker_size = marker_size
        self._edge_width = edge_width
        self._update()
