from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QWidget


class BaseMagicSetting(QWidget):
    """Base class that helps use Magic GUI widgets in the json schema widget
    buider.
    """

    valueChanged = Signal(dict)

    def get_mgui(self):
        """Get magic gui widget."""
        raise NotImplementedError()

    def __init__(self, description=None):
        super().__init__()
        self._widget = self.get_mgui()

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self._widget.native)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._description = description

        @self._widget.changed.connect
        def _call_magic_gui(value: dict = {}):
            self.valueChanged.emit(value())

    def __getattr__(self, attribute: str = None):
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
        """Set tooltip for the magic gui widget.

        Parameters
        ----------
        value: str
            Tooltip for widget.
        """

        if value:
            self._widget.tooltip = value
