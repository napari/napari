from .image_type_enum import ImageType


class NImage():


    def __init__(self, array, name:str = '', type:ImageType = ImageType.Mono):

        self.name     = name
        self.array    = array
        self.type     = type
        self.metadata = {}


    def data_type(self):
        return self.array.dtype

    def is_rgb(self):
        return self.type == ImageType.RGB
