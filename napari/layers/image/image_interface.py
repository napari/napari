from abc import abstractmethod
from ..base._base_layer_interface import BaseLayerInterface


class ImageLayerInterface(BaseLayerInterface):
    """
    Defines the getters/setters editable across components for an ImageLayer
    """

    @abstractmethod
    def _on_interpolation_change(self, value):
        ...

    @abstractmethod
    def _on_contrast_limits_change(self, value):
        ...

    @abstractmethod
    def _on_rendering_change(self, value):
        ...

    @abstractmethod
    def _on_iso_threshold_change(self, value):
        ...

    @abstractmethod
    def _on_attenuation_change(self, value):
        ...

    @abstractmethod
    def _on_gamma_change(self, value):
        ...

    @abstractmethod
    def _on_colormap_change(self, value):
        ...
