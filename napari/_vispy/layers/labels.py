from napari._vispy.layers.image import VispyImageLayer


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, texture_format=None, **kwargs)
