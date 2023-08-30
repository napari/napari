import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.geometry import MeshData
from vispy.visuals.filters import TextureFilter

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.visuals.surface import SurfaceVisual


class VispySurfaceLayer(VispyBaseLayer):
    """Vispy view for the surface layer.

    View is based on the vispy mesh node and uses default values for
    lighting direction and lighting color. More information can be found
    here https://github.com/vispy/vispy/blob/main/vispy/visuals/mesh.py
    """

    def __init__(self, layer) -> None:
        node = SurfaceVisual()
        self._texture_filter = None
        self._light_direction = (-1, 1, 1)
        self._meshdata = None
        super().__init__(layer, node)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.shading.connect(self._on_shading_change)
        self.layer.events.texture.connect(self._on_texture_change)
        self.layer.events.texcoords.connect(self._on_texture_change)

        self.layer.wireframe.events.visible.connect(
            self._on_wireframe_visible_change
        )
        self.layer.wireframe.events.width.connect(
            self._on_wireframe_width_change
        )
        self.layer.wireframe.events.color.connect(
            self._on_wireframe_color_change
        )
        self.layer.normals.face.events.connect(self._on_face_normals_change)
        self.layer.normals.vertex.events.connect(
            self._on_vertex_normals_change
        )

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        vertices = None
        faces = None
        vertex_values = None
        vertex_colors = None
        if len(self.layer._data_view) and len(self.layer._view_faces):
            # Offsetting so pixels now centered
            # coerce to float to solve vispy/vispy#2007
            # reverse order to get zyx instead of xyz
            vertices = np.asarray(
                self.layer._data_view[:, ::-1], dtype=np.float32
            )
            # due to above xyz>zyx, also reverse order of faces to fix
            # handedness of normals
            faces = self.layer._view_faces[:, ::-1]

            values = self.layer._view_vertex_values
            if len(values):
                vertex_values = values

            colors = self.layer._view_vertex_colors
            if len(colors):
                vertex_colors = colors

        # making sure the vertex data is 3D prevents shape errors with
        # attached filters, instead of trying to attach/detach each time
        if vertices is not None and vertices.shape[-1] == 2:
            vertices = np.pad(
                vertices,
                ((0, 0), (0, 1)),
                mode='constant',
                constant_values=0,
            )
        assert vertices is None or vertices.shape[-1] == 3

        self.node.set_data(
            vertices=vertices,
            faces=faces,
            vertex_values=vertex_values,
            vertex_colors=vertex_colors,
        )

        # disable normals in 2D to avoid shape errors
        if self.layer._slice_input.ndisplay == 2:
            self._meshdata = MeshData()
        else:
            self._meshdata = self.node.mesh_data

        self._on_face_normals_change()
        self._on_vertex_normals_change()

        self._on_texture_change()
        self._on_shading_change()

        self.node.update()

        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_texture_change(self):
        """Update or apply the texture filter"""
        # texture images need to be flipped (np.flipud) because of how OpenGL
        # expects the texture data to be ordered in memory we flip them here
        # when setting up the TextureFilter so napari users can load images
        # for textures normally
        # https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        if self.layer._has_texture and self._texture_filter is None:
            self._texture_filter = TextureFilter(
                np.flipud(self.layer.texture),
                self.layer.texcoords,
            )
            self.node.attach(self._texture_filter)
        elif self.layer._has_texture:
            self._texture_filter.texture = np.flipud(self.layer.texture)
            self._texture_filter.texcoords = self.layer.texcoords

        if self._texture_filter is not None:
            self._texture_filter.enabled = self.layer._has_texture
            self.node.update()

    def _on_colormap_change(self):
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            colors = self.layer.colormap.map(
                np.linspace(0, 1, 256) ** self.layer.gamma
            )
            cmap = VispyColormap(colors)
        else:
            cmap = VispyColormap(*self.layer.colormap)
        if self.layer._slice_input.ndisplay == 3:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self):
        self.node.clim = self.layer.contrast_limits

    def _on_gamma_change(self):
        self._on_colormap_change()

    def _on_shading_change(self):
        shading = None if self.layer.shading == 'none' else self.layer.shading
        if not self.node.mesh_data.is_empty():
            self.node.shading = shading
            self._on_camera_move()
        self.node.update()

    def _on_wireframe_visible_change(self):
        self.node.wireframe_filter.enabled = self.layer.wireframe.visible
        self.node.update()

    def _on_wireframe_width_change(self):
        self.node.wireframe_filter.width = self.layer.wireframe.width
        self.node.update()

    def _on_wireframe_color_change(self):
        self.node.wireframe_filter.color = self.layer.wireframe.color
        self.node.update()

    def _on_face_normals_change(self):
        self.node.face_normals.visible = self.layer.normals.face.visible
        if self.node.face_normals.visible:
            self.node.face_normals.set_data(
                self._meshdata,
                length=self.layer.normals.face.length,
                color=self.layer.normals.face.color,
                width=self.layer.normals.face.width,
                primitive='face',
            )

    def _on_vertex_normals_change(self):
        self.node.vertex_normals.visible = self.layer.normals.vertex.visible
        if self.node.vertex_normals.visible:
            self.node.vertex_normals.set_data(
                self._meshdata,
                length=self.layer.normals.vertex.length,
                color=self.layer.normals.vertex.color,
                width=self.layer.normals.vertex.width,
                primitive='vertex',
            )

    def _on_camera_move(self, event=None):
        if (
            event is not None
            and event.type == 'angles'
            and self.layer._slice_input.ndisplay == 3
        ):
            camera = event.source
            # take displayed up and view directions and flip zyx for vispy
            up = np.array(camera.up_direction)[::-1]
            view = np.array(camera.view_direction)[::-1]
            # combine to get light behind the camera on the top right
            self._light_direction = view - up + np.cross(up, view)

        if self.node.shading_filter is not None:
            self.node.shading_filter.light_dir = self._light_direction

    def reset(self, event=None):
        super().reset()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_shading_change()
        self._on_texture_change()
        self._on_wireframe_visible_change()
        self._on_wireframe_width_change()
        self._on_wireframe_color_change()
        self._on_face_normals_change()
        self._on_vertex_normals_change()
