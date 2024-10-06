from typing import Any

import numpy as np

from napari.layers import Layer


class SampleLayer(Layer):
    def __init__(  # type: ignore [no-untyped-def]
        self,
        data: np.ndarray,
        ndim=None,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,  # this should move to future "data source" object.
        experimental_clipping_planes=None,
        metadata=None,
        mode='pan_zoom',
        multiscale=False,
        name=None,
        opacity=1.0,
        projection_mode='none',
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ) -> None:
        if ndim is None:
            ndim = data.ndim
        super().__init__(
            ndim=ndim,
            data=data,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            mode=mode,
            multiscale=multiscale,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )  # type: ignore [no-untyped-call]
        self._data = data
        self.a = 2

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._data = data
        self.events.data(value=data)

    @property
    def _extent_data(self) -> np.ndarray:
        shape = np.array(self.data.shape)
        return np.vstack([np.zeros(len(shape)), shape - 1])

    def _get_ndim(self) -> int:
        return self.ndim

    def _get_state(self) -> dict[str, Any]:
        base_state = self._get_base_state()
        base_state['data'] = self.data
        return base_state

    def _set_view_slice(self) -> None:
        pass

    def _update_thumbnail(self) -> None:
        pass

    def _get_value(self, position: tuple[int, ...]) -> np.ndarray:
        return self.data[position]

    def _post_init(self) -> None:
        self.a = 1
