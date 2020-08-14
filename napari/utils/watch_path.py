import os
import re
import time
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from napari._qt.qthreading import thread_worker
from napari.plugins.io import read_data_with_plugins
from napari.types import FullLayerData


@thread_worker
def path_watcher(path, interval=0.1) -> Generator[Optional[str], None, None]:
    # does not recurse
    files_last = set(os.listdir(path))
    while True:
        files_new = set(os.listdir(path))
        diff = files_new - files_last
        if diff:
            for p in diff:
                print('watcher yielding', p)
                yield os.path.join(path, p)
        else:
            yield None
        files_last = files_new
        time.sleep(interval)


@thread_worker
def path_reader(
    plugin=None,
) -> Generator[Optional[Tuple[str, List[FullLayerData]]], Optional[str], None]:
    from napari.components.add_layers_mixin import _normalize_layer_data

    path = None
    data = None
    while True:
        if path:
            data = read_data_with_plugins(path, plugin=plugin)
            data = (path, [_normalize_layer_data(d) for d in data])
        path = yield data
        data = None
        time.sleep(0.1)


def regex_parser(pattern) -> Callable[[str], Tuple[int, ...]]:
    _parser = re.compile(pattern)

    def parser(path: str) -> Tuple[int, ...]:
        match = _parser.match(path)
        return tuple(int(x) for x in match.groups()) if match else ()

    return parser


class Stacker:
    COLORS = ['green', 'magenta', 'blue']  # whatever...

    def __init__(
        self, viewer, channel_parser: Callable[[str], Tuple[int, ...]] = None
    ):
        self.viewer = viewer
        self.shapes: Dict[Tuple, List[Tuple]] = {}
        self.channel_parser = channel_parser

    def __call__(self, layer_data: Optional[Tuple[str, List[FullLayerData]]]):
        if not layer_data:
            return
        path, _layer_data = layer_data
        _ch = self.channel_parser(path) if self.channel_parser else ()

        assert (_ch == ()) or len(_ch) == 1, 'one channel per image for now'
        assert len(_layer_data) == 1, 'one channel per image for now'

        _ld = _layer_data[0]

        if _ch not in self.shapes:
            try:
                self.shapes[_ch] = [d[0].shape for d in _layer_data]
            except AttributeError:
                raise TypeError(
                    "Stacker only handles numpy-like arrays, "
                    f"got layer data: {_layer_data}"
                )
            _ld[1].setdefault('metadata', {})
            _ld[1]['metadata']['channel'] = _ch
            self.viewer._add_layer_from_data(*_ld)
            if len(self.shapes) == 2:
                # we added a second channel
                for n, i in enumerate(self.shapes):
                    layer = next(
                        lay
                        for lay in self.viewer.layers
                        if lay.metadata.get('channel') == i
                    )
                    layer.colormap = self.COLORS[n]
                    layer.blending = 'additive'
            if len(self.shapes) > 2:
                next(
                    lay
                    for lay in self.viewer.layers
                    if lay.metadata.get('channel') == _ch
                ).colormap = self.COLORS[len(self.shapes)]
                layer.blending = 'additive'
        else:
            img = _ld[0]
            shps = self.shapes[_ch]
            layer = next(
                lay
                for lay in self.viewer.layers
                if lay.metadata.get('channel') == _ch
            )
            # FIXME: only works for single channel per image
            if layer.data.shape == shps[0]:
                # this is the second image in the timelapse
                layer.data = np.stack((layer.data, img), axis=0)
            else:
                layer.data = np.concatenate(
                    (layer.data, img[np.newaxis, :, :]), axis=0
                )

        self.viewer.dims.set_point(0, self.viewer.dims.max_indices[0])


# example channel_parser = regex_parser(r'.*ch(\d{1}).*')
def watch_path(viewer, path, *, plugin=None, channel_parser=None):
    """Watch a path for a new files to concatenate on existing layers.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer instance to which to add files.
    path : str
        The directory to watch
    plugin : str, optional
        A plugin to use when opening files, by default None (will use first
        appropriate plugin)
    channel_parser : callable, optional
        A callable that accepts a path (a filename added to the path) and
        returns a tuple of channel strings found in that filename.  If not
        provided ``watch`` will assume that every file in the watched path
        belongs to a single channel/layer, otherwise, a new layer will be made
        for each returned channel pattern.
    Returns
    -------
    stop : callable
        A function that can be called to stop the watcher
    """
    if not (isinstance(path, str) and os.path.isdir(path)):
        raise ValueError(
            "'watch_path' requires that `path` be an existing directory"
        )
    watcher = path_watcher(path)
    reader = path_reader(plugin)
    stacker = Stacker(viewer, channel_parser)
    watcher.yielded.connect(reader.send)
    reader.yielded.connect(stacker)
    reader.start()
    watcher.start()

    def stop():
        reader.stop()
        watcher.stop()

    return stop
