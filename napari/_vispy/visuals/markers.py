from vispy.scene.visuals import Markers as BaseMarkers

clamp_shader = """
float clamped_size = clamp($v_size, $canvas_size_min, $canvas_size_max);
float clamped_ratio = clamped_size / $v_size;
$v_size = clamped_size;
v_edgewidth = v_edgewidth * clamped_ratio;
gl_PointSize = $v_size + 4. * (v_edgewidth + 1.5 * u_antialias);
"""

old_vshader = BaseMarkers._shaders['vertex']
new_vshader = old_vshader[:-2] + clamp_shader + '\n}'  # very ugly...


class Markers(BaseMarkers):
    _shaders = {
        'vertex': new_vshader,
        'fragment': BaseMarkers._shaders['fragment'],
    }

    def __init__(self, *args, **kwargs) -> None:
        self._canvas_size_limits = 0, 10000
        super().__init__(*args, **kwargs)
        self.canvas_size_limits = 0, 10000

    def _compute_bounds(self, axis, view):
        # needed for entering 3D rendering mode when a points
        # layer is invisible and the self._data property is None
        if self._data is None:
            return None
        pos = self._data['a_position']
        if pos is None:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        else:
            return (0, 0)

    @property
    def canvas_size_limits(self):
        return self._canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        self._canvas_size_limits = value
        self.shared_program.vert['canvas_size_min'] = value[0]
        self.shared_program.vert['canvas_size_max'] = value[1]
