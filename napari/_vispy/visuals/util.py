class TextureMixin:
    """
    This mixin is created to avoid using vispy's private api.
    vispy stores texture format `node._texture.internalformat`
    And we should not relay on it if it is not needed.
    """

    def __init__(self, *args, texture_format, **kwargs):
        super().__init__(*args, texture_format=texture_format, **kwargs)
        # save texture format to not depend on vispy's private api
        self.unfreeze()
        self._texture_format = texture_format
        self.freeze()

    @property
    def texture_format(self):
        return self._texture_format
