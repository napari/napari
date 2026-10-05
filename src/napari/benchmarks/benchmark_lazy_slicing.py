# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import dask.array as da
import numpy as np
import zarr

from napari.components import ViewerModel
from napari.layers import Image
from napari.utils._dask_utils import _DASK_CACHE

from .utils import Skip, SlowMemoryStore

# Each sample must start cold, so setup runs before every timed call and the
# global dask cache is cleared there. See
# https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks
PARAMS = ([0, 0.05], ['zarr', 'dask'])
PARAM_NAMES = ['latency', 'backend']


def _as_backend(array, backend):
    return da.from_zarr(array) if backend == 'dask' else array


class _ZScroll:
    params = PARAMS
    param_names = PARAM_NAMES
    skip_params = Skip(if_in_pr=lambda latency, backend: latency > 0)
    timeout = 300

    def setup(self, latency, backend):
        store = SlowMemoryStore(load_delay=latency)
        data = zarr.zeros(
            shape=(8, 1024, 1024),
            chunks=(1, 256, 256),
            dtype='uint8',
            store=store,
        )
        data[:] = np.random.default_rng(0).integers(
            0, 255, data.shape, dtype='uint8'
        )
        self.viewer = ViewerModel()
        self.viewer.add_image(_as_backend(data, backend))
        # the viewer opens on the middle plane, so start from plane 0 with
        # nothing cached
        self.viewer.dims.set_current_step(0, 0)
        _DASK_CACHE.cache.clear()

    def _scroll(self, steps):
        for z in steps:
            self.viewer.dims.set_current_step(0, z)


class LazyZScrollSuite(_ZScroll):
    """Step through z planes that have not been loaded yet."""

    def time_first_visit(self, latency, backend):
        self._scroll(range(1, 8))

    time_first_visit.number = 1
    time_first_visit.warmup_time = 0


class LazyZScrollRevisitSuite(_ZScroll):
    """Step back through z planes that were already loaded."""

    def setup(self, latency, backend):
        super().setup(latency, backend)
        self._scroll(range(1, 8))

    def time_revisit(self, latency, backend):
        self._scroll(range(6, 0, -1))

    time_revisit.number = 1
    time_revisit.warmup_time = 0


class _MultiscalePan:
    params = PARAMS
    param_names = PARAM_NAMES
    skip_params = Skip(if_in_pr=lambda latency, backend: latency > 0)
    timeout = 300

    def setup(self, latency, backend):
        _DASK_CACHE.cache.clear()
        group = zarr.open_group(SlowMemoryStore(load_delay=latency), mode='w')
        rng = np.random.default_rng(0)
        levels = []
        for i in range(4):
            shape = (4096 >> i,) * 2
            level = group.zeros(
                name=str(i), shape=shape, chunks=(256, 256), dtype='uint8'
            )
            level[:] = rng.integers(0, 255, shape, dtype='uint8')
            levels.append(_as_backend(level, backend))
        self.layer = Image(levels, multiscale=True)
        self._pan_to(0)

    def _pan_to(self, offset):
        """Show a 512x512 canvas over full-resolution pixels at offset."""
        self.layer._update_draw(
            scale_factor=1,
            corner_pixels_displayed=np.array(
                [[offset, offset], [offset + 511, offset + 511]]
            ),
            shape_threshold=(512, 512),
        )


class LazyMultiscalePanSuite(_MultiscalePan):
    """Pan to a region of the finest level that has not been loaded yet."""

    def time_pan_away(self, latency, backend):
        self._pan_to(2048)

    time_pan_away.number = 1
    time_pan_away.warmup_time = 0


class LazyMultiscalePanBackSuite(_MultiscalePan):
    """Pan back to a region of the finest level that was already loaded."""

    def setup(self, latency, backend):
        super().setup(latency, backend)
        self._pan_to(2048)

    def time_pan_back(self, latency, backend):
        self._pan_to(0)

    time_pan_back.number = 1
    time_pan_back.warmup_time = 0


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
