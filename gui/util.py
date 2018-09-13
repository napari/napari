"""Miscellaneous utility functions.
"""


class metadata:
    """Stores metadata as attributes and provides hooks for when
    fields are updated.

    Initializes the same way a dict does.

    Attributes
    ----------
    update_hooks : list of callables (string, any) -> None
        Hooks called when a field is updated with the name
        and value it was set to.
    """
    def __init__(self, *dictp, **data):
        self.__dict__.update(*dictp, **data)
        self.update_hooks = []

    def __setattr__(self, name, value):
        """Calls hooks after the attribute is set.
        """
        super().__setattr__(name, value)
        for hook in self.update_hooks:
            hook(name, value)


def is_multichannel(meta):
    """Determines if an image is RGB after checking its metadata.
    """
    try:
        return meta.itype in ('rgb', 'multi', 'multichannel')
    except AttributeError:
        return False


def guess_multichannel(shape):
    """Guesses if an image is multichannel based on its shape.
    """
    first_dims = shape[:-1]
    last_dim = shape[-1]

    average = sum(first_dims) / len(first_dims)

    if average * .95 - 1 <= last_dim <= average * 1.05 + 1:
        # roughly all dims are the same
        return False

    if last_dim in (3, 4):
        if average > 10:
            return True

    diff = average - last_dim

    return diff > last_dim * 100


_app = None
_GUESS = object()


def imshow(image, meta=None, multichannel=_GUESS, **kwargs):
    """Displays an image.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : metadata, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel.
    **kwargs : dict
        Parameters that will be translated to metadata.

    Returns
    -------
    window : ImageWindow
        Image window. Will auto-delete if out of scope.
    """
    from PyQt5.QtWidgets import QApplication
    from .elements.image_window import ImageWindow

    if meta is None:
        meta = metadata(**kwargs)

    if isinstance(meta, dict):
        meta = metadata(meta, **kwargs)

    if multichannel is _GUESS:
        multichannel = guess_multichannel(image.shape)

    if multichannel:
        meta.itype = 'multi'

    global _app
    _app = QApplication.instance() or QApplication([])

    win = ImageWindow()
    win.add_image(image, meta)

    win.raise_()
    win.show()

    return win
