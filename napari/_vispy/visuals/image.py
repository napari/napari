from vispy.scene.visuals import Image as BaseImage


class Image(BaseImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
