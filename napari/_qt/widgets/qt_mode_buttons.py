import weakref

from qtpy.QtWidgets import QPushButton, QRadioButton


class QtModeRadioButton(QRadioButton):
    """Creates a radio button that can enable a specific layer mode.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer instance that this button controls.
    button_name : str
        Name for the button.  This is mostly used to identify the button
        in stylesheets (e.g. to add a custom icon)
    mode : Enum
        The mode to enable when this button is clicked.
    tooltip : str, optional
        A tooltip to display when hovering the mouse on this button,
        by default it will be set to `button_name`.
    checked : bool, optional
        Whether the button is activate, by default False.
        One button in a QButtonGroup should be initially checked.

    Attributes
    ----------
    layer : napari.layers.Layer
        The layer instance that this button controls.
    """

    def __init__(
        self, layer, button_name, mode, *, tooltip=None, checked=False
    ):
        super().__init__()

        self.layer_ref = weakref.ref(layer)
        self.setToolTip(tooltip or button_name)
        self.setChecked(checked)
        self.setProperty('mode', button_name)
        self.setFixedWidth(28)
        self.mode = mode
        if mode is not None:
            self.toggled.connect(self._set_mode)

    def _set_mode(self, bool):
        """Toggle the mode associated with the layer.

        Parameters
        ----------
        bool : bool
            Whether this mode is currently selected or not.
        """
        layer = self.layer_ref()
        if layer is None:
            return

        with layer.events.mode.blocker(self._set_mode):
            if bool:
                layer.mode = self.mode


class QtModePushButton(QPushButton):
    """Creates a radio button that can trigger a specific action.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer instance that this button controls.
    button_name : str
        Name for the button.  This is mostly used to identify the button
        in stylesheets (e.g. to add a custom icon)
    slot : callable, optional
        The function to call when this button is clicked.
    tooltip : str, optional
        A tooltip to display when hovering the mouse on this button.

    Attributes
    ----------
    layer : napari.layers.Layer
        The layer instance that this button controls.
    """

    def __init__(self, layer, button_name, *, slot=None, tooltip=None):
        super().__init__()

        self.layer = layer
        self.setProperty('mode', button_name)
        self.setToolTip(tooltip or button_name)
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        if slot is not None:
            self.clicked.connect(slot)
