class TextureMixin:
    """Store texture format passed to VisPy classes.

    We need to refer back to the texture format, but VisPy
    stores it in a private attribute — ``node._texture.internalformat``.
    This mixin is added to our Node subclasses to avoid having to
    access private VisPy attributes.
    """

    def __init__(self, *args, texture_format, **kwargs):
        super().__init__(*args, texture_format=texture_format, **kwargs)
        # classes using this mixin may be frozen dataclasses.
        # we save the texture format between unfreeze/freeze.
        self.unfreeze()
        self._texture_format = texture_format
        self.freeze()

    @property
    def texture_format(self):
        return self._texture_format
