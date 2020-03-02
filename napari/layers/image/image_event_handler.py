from typing import List

from ..base._base_event_handler import EventHandlerBase
from .image_interface import ImageInterface


class ImageEventHandler(EventHandlerBase):
    def __init__(self, editable_components: List[ImageInterface]):
        super().__init__(editable_components=editable_components)
