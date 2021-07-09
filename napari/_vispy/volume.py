from vispy.scene.visuals import create_visual_node

from .vendored import VolumeVisual as BaseVolumeVisual

Volume = create_visual_node(BaseVolumeVisual)

# Custom volume class is needed for better 3D rendering
# class Volume(BaseVolume):
#     def __init__(self, *args, **kwargs):
#         self._attenuation = 1.0
#         super().__init__(*args, **kwargs)
#
#     @property
#     def cmap(self):
#         return self._cmap
#
#     @cmap.setter
#     def cmap(self, cmap):
#         self._cmap = get_colormap(cmap)
#         self.shared_program.frag['cmap'] = Function(self._cmap.glsl_map)
#         # Colormap change fix
#         self.shared_program['texture2D_LUT'] = (
#             self.cmap.texture_lut()
#             if (hasattr(self.cmap, 'texture_lut'))
#             else None
#         )
#         self.update()
#
#     @property
#     def threshold(self):
#         """The threshold value to apply for the isosurface render method."""
#         return self._threshold
#
#     @threshold.setter
#     def threshold(self, value):
#         # Fix for #1399, should be fixed in the VisPy threshold setter
#         self._threshold = float(value)
#         self.shared_program['u_threshold'] = self._threshold
#         self.update()
#
#     @property
#     def attenuation(self):
#         """The attenuation value to apply for the attenuated mip render method."""
#         return self._attenuation
#
#     @attenuation.setter
#     def attenuation(self, value):
#         # Fix for #1399, should be fixed in the VisPy threshold setter
#         self._attenuation = float(value)
#         self.shared_program['u_attenuation'] = self._attenuation
#         self.update()
