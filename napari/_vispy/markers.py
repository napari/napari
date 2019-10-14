from vispy.scene.visuals import Markers as BaseMarkers
import numpy as np


class Markers(BaseMarkers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_draw(self, view):
        if self._symbol is None:
            return False
        view.view_program['u_px_scale'] = view.transforms.pixel_scale
        if self.scaling:
            tr = view.transforms.get_transform('visual', 'document').simplified
            mat = tr.map(np.eye(3)) - tr.map(np.zeros((3, 3)))
            scale = np.linalg.norm(mat[:, :3])
            view.view_program['u_scale'] = scale
        else:
            view.view_program['u_scale'] = 1
