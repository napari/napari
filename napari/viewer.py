import zarr
from copy import copy
from ._qt.qt_viewer import QtViewer
from ._qt.qt_main_window import Window
from .components import ViewerModel
from ._version import get_versions
from .util.misc import make_square, set_icon
from imageio import imwrite
import os.path

__version__ = get_versions()['version']
del get_versions


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    file : path or zarr.hierarchy.Group
        Path to napari zarr file or zarr object
    title : string
        The title of the viewer window.
    ndisplay : int
        Number of displayed dimensions.
    tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    """

    _thumbnail_shape = (32, 32, 4)

    def __init__(self, *, file=None, title='napari', ndisplay=2, order=None):
        super().__init__(title=title, ndisplay=ndisplay, order=order)
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.screenshot = self.window.qt_viewer.screenshot
        self.update_console = self.window.qt_viewer.console.push

        if not file is None:
            self.from_zarr(file)

    def to_zarr(self, store=None):
        """Create a zarr group with viewer data."""
        root = zarr.group(store=store)
        root.attrs['napari'] = True
        root.attrs['version'] = __version__
        root.attrs['ndim'] = self.dims.ndim
        root.attrs['ndisplay'] = self.dims.ndisplay
        root.attrs['order'] = [int(o) for o in self.dims.order]
        root.attrs['dims_point'] = [int(p) for p in self.dims.point]
        root.attrs['title'] = self.title
        root.attrs['theme'] = self.theme
        root.attrs['metadata'] = {}
        if self.dims.ndisplay == 3:
            camera = self.window.qt_viewer.view.camera
            camera_dict = {
                'center': camera.center,
                'scale_factor': camera.scale_factor,
                'quaternion': camera._quaternion.get_axis_angle(),
            }
        else:
            r = self.window.qt_viewer.view.camera.rect
            camera_dict = {'rect': [r.left, r.bottom, r.width, r.height]}
        root.attrs['camera'] = camera_dict

        layer_gp = root.create_group('layers')

        layer_names = []
        for layer in self.layers:
            layer_gp = layer.to_zarr(layer_gp)
            layer_names.append(layer.name)
        layer_gp.attrs['layer_names'] = layer_names

        screenshot = self.screenshot()
        root.array(
            'screenshot',
            screenshot,
            shape=screenshot.shape,
            chunks=(None, None, None),
            dtype=screenshot.dtype,
        )

        if store is not None:
            icon = make_square(screenshot)
            imwrite(os.path.join(store, 'screenshot.icns'), icon)
            imwrite(os.path.join(store, 'screenshot.ico'), icon)

            set_icon(store, 'screenshot')

        return root

    def from_zarr(self, root):
        """Load a zarr group with viewer data."""
        if type(root) == str:
            root = zarr.open(root)

        if 'napari' not in root.attrs:
            print('zarr object not recognized as a napari zarr')
            return

        self.dims.ndim = root.attrs['ndim']
        self.dims.ndisplay = root.attrs['ndisplay']
        self.dims.order = root.attrs['order']
        for i, p in enumerate(root.attrs['dims_point']):
            self.dims.set_point(i, p)
        self.title = root.attrs['title']
        self.theme = root.attrs['theme']

        for name in root['layers'].attrs['layer_names']:
            g = root['layers/' + name]
            layer_type = g.attrs['layer_type']
            args = copy(g.attrs.asdict())
            del args['layer_type']
            for array_name, array in g.arrays():
                if array_name not in ['data', 'thumbnail']:
                    args[array_name] = array
            if isinstance(g['data'], zarr.Group):
                data = []
                for array_name, array in g['data'].arrays():
                    data.append(array)
            else:
                data = g['data']
            self._add_layer[layer_type](data, **args)

            camera_dict = root.attrs['camera']
            self.events.reset_view(**camera_dict)
