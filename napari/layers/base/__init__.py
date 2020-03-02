from .base import Layer
from ._base_interface import BaseInterface

from ..image import Image
from ..image.image_event_handler import ImageEventHandler

layer_to_controller = {
    Image: ImageEventHandler,
}
