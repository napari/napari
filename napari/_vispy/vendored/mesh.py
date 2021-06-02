# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""A MeshVisual Visual that uses the new shader Function."""

from __future__ import division

import numpy as np

from vispy.visuals.visual import Visual
from vispy.visuals.shaders import Function, FunctionChain
from vispy.gloo import VertexBuffer
from vispy.geometry import MeshData
from vispy.color import Color, get_colormap
from vispy.util.event import Event


vertex_template = """
varying vec4 v_base_color;

void main() {
    v_base_color = $color_transform($base_color);
    gl_Position = $transform($to_vec4($position));
}
"""

fragment_template = """
varying vec4 v_base_color;
void main() {
    gl_FragColor = v_base_color;
}
"""


# Functions that can be used as is (don't have template variables)
# Consider these stored in a central location in vispy ...

vec3to4 = Function("""
vec4 vec3to4(vec3 xyz) {
    return vec4(xyz, 1.0);
}
""")

vec2to4 = Function("""
vec4 vec2to4(vec2 xyz) {
    return vec4(xyz, 0.0, 1.0);
}
""")


_null_color_transform = 'vec4 pass(vec4 color) { return color; }'
_clim = 'float cmap(float val) { return (val - $cmin) / ($cmax - $cmin); }'


# Eventually this could be de-duplicated with visuals/image.py, which does
# something similar (but takes a ``color`` instead of ``float``)
def _build_color_transform(data, cmap, clim=(0., 1.)):
    if data.ndim == 2 and data.shape[1] == 1:
        fun = Function(_clim)
        fun['cmin'] = clim[0]
        fun['cmax'] = clim[1]
        fun = FunctionChain(None, [fun, Function(cmap.glsl_map)])
    else:
        fun = Function(_null_color_transform)
    return fun


class MeshVisual(Visual):
    """Mesh visual

    Parameters
    ----------
    vertices : array-like | None
        The vertices.
    faces : array-like | None
        The faces.
    vertex_colors : array-like | None
        Colors to use for each vertex.
    face_colors : array-like | None
        Colors to use for each face.
    color : instance of Color
        The color to use.
    vertex_values : array-like | None
        The values to use for each vertex (for colormapping).
    meshdata : instance of MeshData | None
        The meshdata.
    shading : str | None
        Shading to use. This uses the
        :class:`~vispy.visuals.filters.mesh.ShadingFilter`
        filter introduced in VisPy 0.7. This class provides additional
        features that are available when the filter is attached manually.
        See 'examples/basics/scene/mesh_shading.py' for an example.
    mode : str
        The drawing mode.
    **kwargs : dict
        Keyword arguments to pass to `Visual`.

    Notes
    -----
    Additional functionality is available through filters. Mesh-specific
    filters can be found in the :mod:`vispy.visuals.filters.mesh` module.

    This class emits a `data_updated` event when the mesh data is updated. This
    is used for example by filters for synchronization.

    """

    def __init__(self, vertices=None, faces=None, vertex_colors=None,
                 face_colors=None, color=(0.5, 0.5, 1, 1), vertex_values=None,
                 meshdata=None, shading=None, mode='triangles', **kwargs):
        Visual.__init__(self, vcode=vertex_template, fcode=fragment_template,
                        **kwargs)
        self.set_gl_state('translucent', depth_test=True, cull_face=False)

        self.events.add(data_updated=Event)

        # Define buffers
        self._vertices = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._cmap = get_colormap('cubehelix')
        self._clim = 'auto'

        # Uniform color
        self._color = Color(color)

        # add filters for various modifiers
        self.shading_filter = None
        self.shading = shading

        # Init
        self._bounds = None
        # Note we do not call subclass set_data -- often the signatures
        # do no match.
        MeshVisual.set_data(
            self, vertices=vertices, faces=faces, vertex_colors=vertex_colors,
            face_colors=face_colors, vertex_values=vertex_values,
            meshdata=meshdata, color=color)

        # primitive mode
        self._draw_mode = mode

        self.freeze()

    @property
    def shading(self):
        """The shading method."""
        return self._shading

    @shading.setter
    def shading(self, shading):
        assert shading in (None, 'flat', 'smooth')
        self._shading = shading
        if shading is None and self.shading_filter is None:
            # Delay creation of filter until necessary.
            return
        if self.shading_filter is None:
            from .filters import ShadingFilter
            self.shading_filter = ShadingFilter(shading=shading)
        else:
            self.shading_filter.shading = shading
        self.attach(self.shading_filter)

    def set_data(self, vertices=None, faces=None, vertex_colors=None,
                 face_colors=None, color=None, vertex_values=None,
                 meshdata=None):
        """Set the mesh data

        Parameters
        ----------
        vertices : array-like | None
            The vertices.
        faces : array-like | None
            The faces.
        vertex_colors : array-like | None
            Colors to use for each vertex.
        face_colors : array-like | None
            Colors to use for each face.
        color : instance of Color
            The color to use.
        vertex_values : array-like | None
            Values for each vertex.
        meshdata : instance of MeshData | None
            The meshdata.
        """
        if meshdata is not None:
            self._meshdata = meshdata
        else:
            self._meshdata = MeshData(vertices=vertices, faces=faces,
                                      vertex_colors=vertex_colors,
                                      face_colors=face_colors,
                                      vertex_values=vertex_values)
        self._bounds = self._meshdata.get_bounds()
        if color is not None:
            self._color = Color(color)
        self.mesh_data_changed()

    @property
    def clim(self):
        return (self._clim if isinstance(self._clim, str) else
                tuple(self._clim))

    @clim.setter
    def clim(self, clim):
        if isinstance(clim, str):
            if clim != 'auto':
                raise ValueError('clim must be "auto" if a string')
        else:
            clim = np.array(clim, float)
            if clim.shape != (2,):
                raise ValueError('clim must have two elements')
        self._clim = clim
        self.mesh_data_changed()

    @property
    def _clim_values(self):
        if isinstance(self._clim, str):  # == 'auto'
            if self._meshdata.has_vertex_value():
                clim = self._meshdata.get_vertex_values()
                clim = (np.min(clim), np.max(clim))
            else:
                clim = (0, 1)
        else:
            clim = self._clim
        return clim

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = get_colormap(cmap)
        self.mesh_data_changed()

    @property
    def mode(self):
        """The triangle mode used to draw this mesh.

        Options are:

            * 'triangles': Draw one triangle for every three vertices
              (eg, [1,2,3], [4,5,6], [7,8,9)
            * 'triangle_strip': Draw one strip for every vertex excluding the
              first two (eg, [1,2,3], [2,3,4], [3,4,5])
            * 'triangle_fan': Draw each triangle from the first vertex and the
              last two vertices (eg, [1,2,3], [1,3,4], [1,4,5])
        """
        return self._draw_mode

    @mode.setter
    def mode(self, m):
        modes = ['triangles', 'triangle_strip', 'triangle_fan']
        if m not in modes:
            raise ValueError("Mesh mode must be one of %s" % ', '.join(modes))
        self._draw_mode = m

    @property
    def mesh_data(self):
        """The mesh data"""
        return self._meshdata

    @property
    def color(self):
        """The uniform color for this mesh"""
        return self._color

    @color.setter
    def color(self, c):
        """Set the uniform color of the mesh

        This value is only used if per-vertex or per-face colors are not
        specified.

        Parameters
        ----------
        c : instance of Color
            The color to use.
        """
        if c is not None:
            self._color = Color(c)
        self.mesh_data_changed()

    def mesh_data_changed(self):
        self._data_changed = True
        self.update()

    def _update_data(self):
        md = self.mesh_data

        v = md.get_vertices(indexed='faces')
        if v is None:
            return False
        if v.shape[-1] == 2:
            v = np.concatenate((v, np.zeros((v.shape[:-1] + (1,)))), -1)
        self._vertices.set_data(v, convert=True)
        if md.has_vertex_color():
            colors = md.get_vertex_colors(indexed='faces')
            colors = colors.astype(np.float32)
        elif md.has_face_color():
            colors = md.get_face_colors(indexed='faces')
            colors = colors.astype(np.float32)
        elif md.has_vertex_value():
            colors = md.get_vertex_values(indexed='faces')
            colors = colors.ravel()[:, np.newaxis]
            colors = colors.astype(np.float32)
        else:
            colors = self._color.rgba

        self.shared_program.vert['position'] = self._vertices

        self.shared_program['texture2D_LUT'] = self._cmap.texture_lut() \
            if (hasattr(self._cmap, 'texture_lut')) else None

        # Position input handling
        if v.shape[-1] == 2:
            self.shared_program.vert['to_vec4'] = vec2to4
        elif v.shape[-1] == 3:
            self.shared_program.vert['to_vec4'] = vec3to4
        else:
            raise TypeError("Vertex data must have shape (...,2) or (...,3).")

        # Set the base color.
        #
        # The base color is mixed further by the material filters for texture
        # or shading effects.
        self.shared_program.vert['color_transform'] = \
            _build_color_transform(colors, self._cmap, self._clim_values)
        if colors.ndim == 1:
            self.shared_program.vert['base_color'] = colors
        else:
            self.shared_program.vert['base_color'] = VertexBuffer(colors)

        self._data_changed = False

        self.events.data_updated()

    def _prepare_draw(self, view):
        if self._data_changed:
            if self._update_data() is False:
                return False
            self._data_changed = False

    def draw(self, *args, **kwds):
        Visual.draw(self, *args, **kwds)

    @staticmethod
    def _prepare_transforms(view):
        tr = view.transforms.get_transform()
        view.view_program.vert['transform'] = tr

    def _compute_bounds(self, axis, view):
        if self._bounds is None:
            return None
        if axis >= len(self._bounds):
            return (0, 0)
        else:
            return self._bounds[axis]