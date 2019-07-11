from enum import Enum

from vispy.scene import (
    TurntableCamera,
    PanZoomCamera,
    FlyCamera,
    ArcballCamera,
)


class Camera(Enum):
    """Camra: Vispy Camera mode.

    The spatial filters used for camera are from vispy

    https://github.com/vispy/vispy/blob/master/vispy/scene/cameras/__init__.py
    """

    TURNTABLE = TurntableCamera()
    ARCBALL = ArcballCamera()
    FLY = FlyCamera()
    PANZOOM = PanZoomCamera()
