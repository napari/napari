from typing import ClassVar

import numpy as np
from vispy.scene.visuals import Image as BaseImage

from napari._vispy.visuals.util import TextureMixin
from napari.utils.colormaps.colormap_utils import NAN_SENTINEL

_FLT_MAX = float(np.finfo(np.float32).max)

# A compiler that assumes no NaN (e.g. Metal's default fast math) folds any NaN
# test against a single bound, including vispy's `!(d <= 0.0 || 0.0 <= d)`.
# Two bounds it cannot relate survive, so `$flt_max` must stay a uniform. NaN
# continues as NAN_SENTINEL because the colormap's own NaN test folds too.
_APPLY_CLIM_FLOAT = f"""
    float apply_clim(float data) {{
        if (!(data <= $flt_max) && !(data >= -$flt_max)) return {NAN_SENTINEL};

        data = clamp(data, min($clim.x, $clim.y), max($clim.x, $clim.y));
        data = (data - $clim.x) / ($clim.y - $clim.x);
        return data;
    }}"""

_APPLY_GAMMA_FLOAT = f"""
    float apply_gamma(float data) {{
        // pow() of a negative base is NaN, so the sentinel bypasses it.
        if (data < {NAN_SENTINEL / 2}) return data;
        return pow(data, $gamma);
    }}"""


# If data is not present, we need bounds to be None (see napari#3517)
class Image(TextureMixin, BaseImage):
    _func_templates: ClassVar[dict[str, str]] = {
        **BaseImage._func_templates,
        'clim_float': _APPLY_CLIM_FLOAT,
        'gamma_float': _APPLY_GAMMA_FLOAT,
    }

    def _build_color_transform(self):
        chain = super()._build_color_transform()
        for func in chain.functions:
            if 'flt_max' in func.template_vars:
                func['flt_max'] = _FLT_MAX
        return chain

    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        if axis > 1:
            return (0, 0)

        return (0, self.size[axis])
