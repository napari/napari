from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vispy.visuals.visual import Visual
else:

    class Visual:
        pass


class TextureMixin(Visual):
    """Store texture format passed to VisPy classes.

    We need to refer back to the texture format, but VisPy
    stores it in a private attribute — ``node._texture.internalformat``.
    This mixin is added to our Node subclasses to avoid having to
    access private VisPy attributes.
    """

    def __init__(self, *args, texture_format: Optional[str], **kwargs) -> None:  # type: ignore [no-untyped-def]
        super().__init__(*args, texture_format=texture_format, **kwargs)
        # classes using this mixin may be frozen dataclasses.
        # we save the texture format between unfreeze/freeze.
        self.unfreeze()
        self._texture_format = texture_format
        self.freeze()

    @property
    def texture_format(self) -> Optional[str]:
        return self._texture_format
