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


def is_rgb(meta):
    return meta.itype == 'rgb'
