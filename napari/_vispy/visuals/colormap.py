import numpy as np
from vispy.scene import Image, Node, STTransform


class Colormap(Node):
    def __init__(self) -> None:
        super().__init__()
        self.img = Image(parent=self)
        self.img.transform = STTransform(scale=(50, 1))
        self.set_data_and_clim()

    def set_data_and_clim(self, clim=(0, 1), dtype=np.float32):
        self.img.set_data(
            np.linspace(clim[1], clim[0], 250, dtype=dtype).reshape(-1, 1)
        )
        self.img.clim = clim

    def set_cmap(self, cmap):
        self.img.cmap = cmap

    def set_gamma(self, gamma):
        self.img.gamma = gamma

    def set_gl_state(self, *args, **kwargs):
        self.img.set_gl_state(*args, **kwargs)
