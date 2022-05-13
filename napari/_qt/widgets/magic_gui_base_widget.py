from qtpy.QtCore import QObject, Signal


class BaseMagicSetting(QObject):
    """Base class that helps use Magic GUI widgets in the json schema widget
    buider.
    """

    MAGIC_GUI = None
    valueChanged = Signal(dict)

    def __init__(self, description=None):
        super().__init__()
        self._widget = self.__class__.MAGIC_GUI
        self._description = description
        self._widget.changed.connect(
            lambda _: self.valueChanged.emit(_.value())
        )

    def __getattr__(self, attribute):
        """Method that will retrieve needed information
        from the actual widget if not found in the class itself.
        """

        try:
            return self.__getattribute__(attribute)
        except AttributeError:
            try:
                return getattr(self._widget.native, attribute)
            except AttributeError as e:
                raise (e)

    def setDescription(self, value):
        """Set description of magic gui widget.

        Parameters
        ----------
        value: str
            Description for widget.

        """

        self._description = value

    def setToolTip(self, value):
        """Set tooltip for each widget in the magic gui widget.

        Note: Currently, there is only 1 tooltip that will be used
        for all widgets.

        Parameters
        ----------
        value: str
            Tooltip for widget.

        """

        for wdg in self._widget._list:
            wdg.tooltip = value
