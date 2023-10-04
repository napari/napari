import logging
import sys
import numpy as np
import dask.array as da

from typing import Dict, Iterable, List, Optional, Tuple, Union

from napari.layers._data_protocols import Index, LayerDataProtocol

LOGGER = logging.getLogger("napari.experimental._virtual_data")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


class VirtualData:
    """`VirtualData` wraps a particular scale level's array.

    It acts like an array of that size, but only works within the interval
    setup by `set_interval`. Each `VirtualData` uses the scale level's
    coordinates.

    -- `VirtualData` uses a hyperslice to store the currently active interval.
    -- `VirtualData.translate` specifies the offset of the
    `VirtualData.hyperslice`'s origin from `VirtualData.array`'s origin, in
    `VirtualData`'s coordinate system

    VirtualData is used to use a ND array to represent
    a larger shape. The purpose of this function is to provide
    a ND slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last `ndisplay` dimensions
    will be used as the (y, x) values.

    NEW: use a translate to define subregion of image

    Attributes
    ----------
    array: ndarray-like
        nd-array like probably a Dask array or a zarr Array
    dtype: dtype
        dtype of the array
    shape: tuple
        shape of the true data (not the hyperslice)
    ndim: int
        Number of dimensions for this scale
    translate: list[tuple(int)]
        tuple for the translation
    d: int
        Dimension of the chunked slices ??? Hard coded to 2.
    hyperslice: dask.array
        Array of currently visible data for this layer
    _min_coord: list
        List of the minimum coordinates in each dimension
    _max_coord: list
        List of the maximum coordinates in each dimension

    """

    def __init__(self, array, scale, ndisplay=2):
        self.array = array
        self.dtype = array.dtype
        # This shape is the shape of the true data, but not our hyperslice
        self.shape = array.shape
        self.ndim = len(self.shape)

        self.translate = tuple([0] * len(self.shape))

        self.hyperslice = da.zeros(1)
        self.ndisplay = ndisplay

        self._max_coord = None
        self._min_coord = None
        self.scale = scale  # for debugging

    def set_interval(self, coords):
        """Interval is the data for this scale that is currently visible.

        It is stored in `_min_coord` and `_max_coord` in the coordinates of
        the original array.

        This function takes a slice, converts it to a range (stored as
        `_min_coord` and `_max_coord`), and extracts a subset of the original
        data array (stored as `hyperslice`)

        Parameters
        ----------
        coords: tuple(slice(ndim))
            tuple of slices in the same coordinate system as the parent array.
        """

        # Validate coords
        if not isinstance(coords, tuple):
            raise ValueError("coords must be a tuple of slices.")

        if len(coords) != self.ndim:
            raise ValueError(
                f"coords must have {self.ndim} slices, but got {len(coords)}"
            )

        for i, sl in enumerate(coords):
            if not isinstance(sl, slice):
                raise ValueError(f"coords[{i}] is not a slice object: {sl}")

        if sl.start < 0 or sl.stop > self.shape[i]:
            raise ValueError(
                f"coords[{i}] is out of bounds for array shape: {self.shape}"
            )

        # store the last interval
        prev_max_coord = self._max_coord
        prev_min_coord = self._min_coord

        # extract the coordinates as a range
        self._max_coord = [sl.stop for sl in coords]
        self._min_coord = [sl.start for sl in coords]

        # Round max and min coord according to self.array.chunks
        # for each dimension, reset the min/max coords, aka interval to be
        # the range of chunk coordinates since we can't partially load a chunk
        for dim in range(len(self._max_coord)):
            if isinstance(self.array, da.Array):
                chunks = self.array.chunks[dim]
                cumuchunks = np.array(chunks).cumsum()
            else:
                # For zarr
                cumuchunks = list(
                    range(
                        self.chunks[dim],
                        self.array.shape[dim],
                        self.chunks[dim],
                    )
                )
                # Add last element
                cumuchunks += [self.array.shape[dim]]
                cumuchunks = np.array(cumuchunks)

            # First value greater or equal to
            min_where = np.where(cumuchunks >= self._min_coord[dim])
            if min_where[0].size == 0:
                breakpoint()
            greaterthan_min_idx = (
                min_where[0][0] if min_where[0] is not None else 0
            )
            self._min_coord[dim] = (
                cumuchunks[greaterthan_min_idx - 1]
                if greaterthan_min_idx > 0
                else 0
            )

            max_where = np.where(cumuchunks >= self._max_coord[dim])
            if max_where[0].size == 0:
                breakpoint()
            lessthan_max_idx = (
                max_where[0][0] if max_where[0] is not None else 0
            )
            self._max_coord[dim] = (
                cumuchunks[lessthan_max_idx]
                if lessthan_max_idx < cumuchunks[-1]
                else cumuchunks[-1] - 1
            )

        # Update translate
        self.translate = self._min_coord

        # interval size may be one or more chunks
        interval_size = [
            mx - mn for mx, mn in zip(self._max_coord, self._min_coord)
        ]

        LOGGER.debug(
            f"VirtualData: update_with_minmax: {self.translate} max \
            {self._max_coord} interval size {interval_size}"
        )

        # Update hyperslice
        new_shape = [
            int(mx - mn) for (mx, mn) in zip(self._max_coord, self._min_coord)
        ]

        # Try to reuse the previous hyperslice if possible (otherwise we get
        # flashing) shape of the chunks
        next_hyperslice = np.zeros(new_shape, dtype=self.dtype)

        if prev_max_coord:
            # Get the matching slice from both data planes
            next_slices = []
            prev_slices = []
            for dim in range(len(self._max_coord)):
                # to ensure that start is non-negative
                # prev_start is the start of the overlapping region in the
                # previous one
                if self._min_coord[dim] < prev_min_coord[dim]:
                    prev_start = 0
                    next_start = prev_min_coord[dim] - self._min_coord[dim]
                else:
                    prev_start = self._min_coord[dim] - prev_min_coord[dim]
                    next_start = 0

                width = min(
                    self.hyperslice.shape[dim], next_hyperslice.shape[dim]
                )
                # to make sure its not overflowing the shape
                width = min(
                    width,
                    width
                    - ((next_start + width) - next_hyperslice.shape[dim]),
                    width
                    - ((prev_start + width) - self.hyperslice.shape[dim]),
                )

                prev_stop = prev_start + width
                next_stop = next_start + width

                LOGGER.debug(
                    f"Dimension {dim}, Prev start: {prev_start}, Next start: {next_start}, Width: {width}"
                )

                prev_slices += [slice(int(prev_start), int(prev_stop))]
                next_slices += [slice(int(next_start), int(next_stop))]

            if (
                next_hyperslice[tuple(next_slices)].size > 0
                and self.hyperslice[tuple(prev_slices)].size > 0
            ):
                LOGGER.info(
                    f"reusing data plane: prev {prev_slices} next \
                    {next_slices}"
                )
                if (
                    next_hyperslice[tuple(next_slices)].size
                    != self.hyperslice[tuple(prev_slices)].size
                ):
                    import pdb

                    pdb.set_trace()
                next_hyperslice[tuple(next_slices)] = self.hyperslice[
                    tuple(prev_slices)
                ]
            else:
                LOGGER.info(
                    f"could not reuse data plane: prev {prev_slices} next \
                    {next_slices}"
                )

        self.hyperslice = next_hyperslice

    def _hyperslice_key(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ):
        """Convert a key into a key for the hyperslice.

        The _hyperslice_key is the corresponding value of key
        within plane. The difference between key and hyperslice_key is
        key - translate / scale
        """
        # TODO maybe fine, but a little lazy
        if type(key) is not tuple:
            key = (key,)

        indices = [None] * len(key)
        for i, index in enumerate(indices):
            if type(index) == slice:
                indices[i] = slice(
                    index.start - self.translate[i],
                    index.stop - self.translate[i],
                )
            elif np.issubdtype(int, type(index)):
                indices[i] = index - self.translate[i]
            # else:
            #     LOGGER.info(f"_hyperslice_key: unexpected type {type(index)}\
            #     with value {index}")

        if type(key) is tuple and type(key[0]) == slice:
            if key[0].start is None:
                return key
            hyperslice_key = tuple(
                [
                    slice(
                        int(
                            max(
                                0,
                                sl.start - self.translate[idx],
                            )
                        ),
                        int(
                            max(
                                0,
                                sl.stop - self.translate[idx],
                            )
                        ),
                        sl.step,
                    )
                    for (idx, sl) in enumerate(key[(-self.ndisplay) :])
                ]
            )
        elif type(key) is tuple and type(key[0]) is int:
            hyperslice_key = tuple(
                [
                    int(v - self.translate[idx])
                    for idx, v in enumerate(key[(-self.ndisplay) :])
                ]
            )
        else:
            LOGGER.info(f"_hyperslice_key: funky key {key}")
            hyperslice_key = key

        return hyperslice_key

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Return item from array.

        key is in data coordinates
        """
        return self.get_offset(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Set an item in the array.

        key is in data coordinates
        """
        self.hyperslice[key] = value
        return self.hyperslice[key]

    def get_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Return item from array.

        key is in data coordinates.
        """
        hyperslice_key = self._hyperslice_key(key)
        try:
            return self.hyperslice[
                hyperslice_key
            ]  # Using square bracket notation
        except Exception:
            # if it isn't a slice it is an int so width=1
            shape = tuple(
                [
                    (1 if sl.start is None else sl.stop - sl.start)
                    if type(sl) is slice
                    else 1
                    for sl in key
                ]
            )
            LOGGER.info(f"get_offset failed {key}")
            return np.zeros(shape)

    def set_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Return self[key]."""
        hyperslice_key = self._hyperslice_key(key)
        LOGGER.info(
            f"hyperslice_key {hyperslice_key} hyperslice shape {self.hyperslice.shape}"
        )
        LOGGER.info(
            f"set_offset: {hyperslice_key} hyperslice shape: \
            {self.hyperslice[hyperslice_key].shape} value shape: {value.shape} "
        )

        # TODO hack for malformed data
        if not np.all(
            np.array(value.shape)
            == np.array(self.hyperslice[hyperslice_key].shape)
        ):
            # Transpose value to match the expected shape
            value = np.transpose(value, axes=(2, 1, 0))

        if self.hyperslice[hyperslice_key].size > 0:
            self.hyperslice[hyperslice_key] = value
        return self.hyperslice[hyperslice_key]

    @property
    def chunksize(self):
        """Return the size of a chunk."""
        if isinstance(self.array, da.Array):
            return self.array.chunksize
        else:
            return self.array.info  # Based on zarr

    @property
    def chunks(self):
        """Return the chunks of the array."""
        # TODO this isn't safe because array can be dask or zarr
        return self.array.chunks


class MultiScaleVirtualData:
    """MultiScaleVirtualData encapsulates multiscale arrays.

    The added value is that MultiScaleVirtualData has a small memory
    footprint.

    MultiScaleVirtualData is a parent that tracks the transformation
    between each VirtualData child relative to the highest resolution
    VirtualData (which is just a re-scaling transform). It accepts inputs in
    the coordinate system of the highest resolution VirtualData.

    VirtualData is used to use a 2D or 3D subarray to represent
    a larger shape. The purpose of this function is to provide
    a 2D or 3D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.

    _set_interval must be called to initialize

    NEW: use a translate to define subregion of image

    Attributes
    ----------
    arrays: dask.array
        List of dask arrays of image data, one for each scale
    dtype: dtype
        dtype of the arrays
    shape: tuple
        Shape of the true data at the highest resolution (x, y) or (x, y, z)
    ndim: int
        Number of dimensions, aka scales
    _data: list[VirtualData]
        List of VirtualData objects for each scale
    _translate: list[tuple(int)]
        List of tuples, e.g. [(0, 0), (0, 0)] of translations for each scale
        array
    _scale_factors: list[list[float]]
        List of scale factors between the highest resolution scale and each
        subsequent scale
    ndisplay: int
        number of dimensions to display (equivalent to ndisplay in
        napari.viewer.Viewer)
    _chunk_slices: list of list of chunk slices
        List of list of chunk slices. A slice object for each scale, for each
        dimension,
        e.g. [list of scales[list of slice objects[(x, y, z)]]]
    """

    def __init__(self, arrays, ndisplay=2):
        # TODO arrays should be typed as MultiScaleData
        # each array represents a scale
        self.arrays = arrays
        # first array is considered the highest resolution
        # TODO [kcp] - is that always true?
        highest_res = arrays[0]

        self.dtype = highest_res.dtype
        # This shape is the shape of the true data, but not our hyperslice
        self.shape = highest_res.shape

        self.ndim = len(arrays)

        self.ndisplay = ndisplay

        # Keep a VirtualData for each array
        self._data = []
        self._translate = []
        self._scale_factors = []
        for scale in range(len(self.arrays)):
            virtual_data = VirtualData(
                self.arrays[scale], scale=scale, ndisplay=self.ndisplay
            )
            self._translate += [
                tuple([0] * len(self.shape))
            ]  # Factors to shift the layer by in units of world coordinates.
            # TODO [kcp] there are assumptions here, expect rounding errors
            #      here, or should we force ints?
            self._scale_factors += [
                [
                    hr_val / this_val
                    for hr_val, this_val in zip(
                        highest_res.shape, self.arrays[scale].shape
                    )
                ]
            ]
            self._data += [virtual_data]

    def set_interval(self, min_coord, max_coord, visible_scales=None):
        """Set the extents for each of the scales.

        Extents are set in the coordinates of each individual scale array

        visible_scales must be an empty list of a list equal to the number of
        scales

        Sets the _min_coord and _max_coord on each individual VirtualData

        min_coord and max_coord are in the same units as the highest resolution
        scale data.

        Note: if you are using this for stuff that goes to GPU or some other
        memory constrained situation, then be careful about your interval size.

        Parameters
        ----------
        min_coord: np.array
            min coordinate in data space, should correspond to top left corner
            of the visible canvas
        max_coord: np.array
            max coordinate in data space, should correspond to bottom right
            corner of the visible canvas
        visible_scales: list
            Optional. ???
        """
        # Bound min_coord and max_coord
        if visible_scales is None:
            visible_scales = [True] * len(self.arrays)

        max_coord = np.min((max_coord, self._data[0].shape), axis=0)
        min_coord = np.max((min_coord, np.zeros_like(min_coord)), axis=0)

        # for each scale, set the interval for the VirtualData
        # e.g. a high resolution scale may cover [0,1,2,3,4] but a scale
        # with half of that resolution will cover the same region with
        # coords/indices of [0,1,2]
        for scale in range(len(self.arrays)):
            if visible_scales[scale]:
                scaled_min = [
                    int(coord / factor)
                    for coord, factor in zip(
                        min_coord, self._scale_factors[scale]
                    )
                ]
                scaled_max = [
                    int(coord / factor)
                    for coord, factor in zip(
                        max_coord, self._scale_factors[scale]
                    )
                ]

                self._translate[scale] = scaled_min
                LOGGER.info(
                    f"MultiscaleVirtualData: update_with_minmax: scale {scale} min {min_coord} : {scaled_min} max {max_coord} : scaled max {scaled_max}"
                )

                coords = tuple(
                    slice(mn, mx) for mn, mx in zip(scaled_min, scaled_max)
                )
                self._data[scale].set_interval(coords)

