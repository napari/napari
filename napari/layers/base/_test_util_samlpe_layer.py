import numpy as np

from napari.layers import Layer


class SampleLayer(Layer):

    def __init__(
        self,
        data,
        ndim,
        *,
        affine=None,
        axes_labels=None,
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
    ):
        super().__init__(
            ndim=ndim,
            data=data,
            affine=affine,
            axes_labels=axes_labels,
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
        )
        self._data = data
        self.a = 2

    @property
    def data(self):
        return self._data

    @property
    def _extent_data(self) -> np.ndarray:
        pass

    def _get_ndim(self) -> int:
        return self.ndim

    def _get_state(self):
        base_state = self._get_base_state()
        base_state['data'] = self.data
        return base_state

    def _set_view_slice(self):
        pass

    def _update_thumbnail(self):
        pass

    def _get_value(self, position):
        return self.data[position]

    def _post_init(self):
        self.a = 1
