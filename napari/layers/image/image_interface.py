from abc import abstractmethod
from ..base._base_interface import BaseInterface


class ImageInterface(BaseInterface):
    """
    Defines the getters/setters editable across components for an ImageLayer
    """

    @abstractmethod
    def _set_interpolation(self, value):
        ...

    @abstractmethod
    def _set_contrast_limits(self, value):
        ...
