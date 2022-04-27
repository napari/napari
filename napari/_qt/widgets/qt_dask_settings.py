from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from ...utils.translations import trans


class QtDaskSettingsWidget(QWidget):

    """Creates custom widget to set enable/disable dask and set cache size.

    Parameters
    ----------
    description : str
        Text to explain and display on widget.
    enabled: bool
        Bool value indicating if dask is enabled.
    value : dict
        value = {'enabled': dask enabled (True/False),
        'cache': cache size in bytes}
    cache : int
        Dask cache size in bytes.
    min_value : int
        Minimum value of allowable cache range.
    max_value : int
        Maximum value of allowable cache range.
    inc : int
        Increment of cache step for cache value widget (in bytes).
    """

    valueChanged = Signal(dict)

    def __init__(
        self,
        parent: QWidget = None,
        description: str = "",
        enabled: bool = True,
        value: dict = None,
        cache: int = 1,
        min_value: int = 1,
        max_value: int = 10,
        inc: int = 1,
    ):
        super().__init__()

        self._cache_value = cache
        self._min_value = min_value
        self._max_value = max_value
        self._enabled = enabled
        self._description = description
        self._value = value
        self._increment = inc

        # Widget
        self._enabled_checkbox = QCheckBox(self)
        self._label = QLabel(self)
        self._units = QLabel(self)
        self._cache = QSpinBox(self)

        self._cache.setMinimum(min_value)
        self._cache.setMaximum(max_value)
        self._cache.setValue(cache)
        self._cache.setSingleStep(inc)
        self._enabled_checkbox.setChecked(enabled)
        self._label.setText('Cache size: ')
        self._units.setText(f'/{self._max_value}')
        self._cache.setDisabled(not enabled)

        # Signals
        self._enabled_checkbox.stateChanged.connect(self._on_enabled_checkbox)

        self._cache.valueChanged.connect(self._update_cache)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self._enabled_checkbox)
        layout.addWidget(self._label)
        layout.addWidget(self._cache)
        layout.addWidget(self._units)

        self.setLayout(layout)

    def _update_cache(self, value):
        '''Update dask cache value and emits signal that value was changed.

        Parameters
        ----------
        value: int
            New dask cache value in bytes.

        '''
        self._cache_value = value
        self.valueChanged.emit(self.value())

    def _on_enabled_checkbox(self, event):

        '''Method that updates dask enabled/disabled status.
        Emits signal when status is changed.

        Parameters
        ----------
        event: bool
            Indicates if dask is enabled (True) or not (False).

        '''

        if event:
            self._cache.setDisabled(False)
            self._enabled = True
        else:
            self._cache.setDisabled(True)
            self._enabled = False

        self.valueChanged.emit(self.value())

    def _update_units(self, value):
        '''Update string displayed next to cache value.

        Parameters
        ----------
        value: str
            String after cach value that consists of max allowable cache in mb.
                f'/{self._max_value} mb'

        '''
        self._units.setText(value)

    def setDescription(self, value):
        '''Set description of dask settings widget.

        Parameters
        ----------
        value: str
            Description for dask settings widget.
        '''
        self._description = value

    def value(self):
        """Return current dask cache value.

        Returns
        -------
        value: dict
            Current value of dask widget.
            {'enabled': self._enabled, 'cache': self._cache_value*1000000}
            enabled: bool
            cache: int (bytes)
        """
        value = {
            'enabled': self._enabled,
            'cache': self._cache_value * 1000000,
        }
        return value

    def setValue(self, value):
        """Set new value and update widget.

        Parameters
        ----------
        value : dict
            Dask cache value.
            value = {'enabled': self._enabled, 'cache': self._cache_value*1000000}
            enabled: bool
            cache: int (bytes)
        """

        if value == "":
            return

        if value == {}:
            return

        cache = int(value['cache'] / 1000000)
        cache = max(min(cache, self._max_value), self._min_value)
        if value == self._value:
            return

        value['cache'] = cache
        self._value = value
        self._cache_value = value['cache']
        self._cache.setValue(value['cache'])
        self._enabled
        self._enabled = value['enabled']
        self._enabled_checkbox.setChecked(value['enabled'])

    def setMinimum(self, value):
        """Set minimum dask cache value for spinbox widget.

        Parameters
        ----------
        value : int
            Minimum dask cache value in bytes.
        """
        value = int(value / 1000000)
        if value < self._max_value:
            self._min_value = value
            self._triangle.setMinimum(value)
            self._cache_value = (
                self._min_value
                if self._cache_value < self._min_value
                else self._cache_value
            )
            self._cache.setMinimum(value)
        else:
            raise ValueError(
                trans._(
                    "Minimum value must be smaller than {max_value}",
                    deferred=True,
                    max_value=self._max_value,
                )
            )

    def minimum(self):
        """Return minimum dask cache value.

        Returns
        -------
        int
            Minimum value of cache widget in bytes.
        """
        return self._min_value * 1000000

    def setIncrement(self, value):
        """Set increment to adjust dask cache value on widget.

        Parameters
        ----------
        value : int
            Increment/step size for dask cache in mb.
        """

        value = int(value)

        if value > (self._max_value - self._min_value):
            return

        if value < 1:
            return

        self._increment = value
        self._cache.setSingleStep(value)

    def setMaximum(self, value):
        """Set maximum dask cache value.

        Parameters
        ----------
        value : int
            Maximum dask cache value in bytes.
        """
        value = int(value / 1000000)
        if value > self._min_value:
            self._max_value = value
            self._cache_value = (
                self._max_value
                if self._cache_value > self._max_value
                else self._cache_value
            )
            self._cache.setMaximum(value)
            self._update_units(f'/{self._max_value} mb')
        else:
            raise ValueError(
                trans._(
                    "Maximum value must be larger than {min_value}",
                    deferred=True,
                    min_value=self._min_value,
                )
            )

    def maximum(self):
        """Return maximum dask cache value.

        Returns
        -------
        int
            Maximum value of dask cache in bytes.
        """
        return self._max_value * 1000000
