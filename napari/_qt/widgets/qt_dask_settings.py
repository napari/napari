# import sys

from magicgui import magicgui
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from napari.utils.translations import trans
<<<<<<< HEAD
=======

# from napari.utils.events.custom_types import conint

# from ...utils.translations import trans


@magicgui(auto_call=True, layout='horizontal', cache={'min': 0, 'max': 20})
def dask_settings(dask_enabled=True, cache=15.0) -> int:
    return {'enabled': dask_enabled, 'cache': cache}


class DaskSettings(QObject):

    valueChanged = Signal(dict)

    def __init__(self, description=None):
        super().__init__()
        self.widget = dask_settings
        self._description = description

        self.widget.changed.connect(
            lambda _: self.valueChanged.emit(_.value())
        )

    def setDescription(self, value):
        '''Set description of dask settings widget.

        Parameters
        ----------
        value: str
            Description for dask settings widget.
        '''
        self._description = value

    def setToolTip(self, value):
        self.widget.dask_enabled.tooltip = value
        self.widget.cache.tooltip = value
>>>>>>> f495a923 (change to magic gui widget)


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
        'cache': cache size in mb}
    cache : int
        Dask cache size in mb.
    min_value : int
        Minimum value of allowable cache range.
    max_value : int
        Maximum value of allowable cache range.
    inc : int
        Increment of cache step for cache value widget (in mb).
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
        unit: str = 'mb',
    ):
        super().__init__()

        if value is not None:
            cache = value['cache']
            enabled = value['enabled']

        self._min_value = min_value
        self._max_value = max_value
        if cache <= max_value:
            if cache >= min_value:
                self._cache_value = cache
            else:
                self._cache_value = min_value
        else:
            self._cache_value = max_value

        self._enabled = enabled
        self._description = description
        self._value = value
        self._increment = inc
        self._unit = unit

        # Widget
        self._enabled_checkbox = QCheckBox(self)
        self._label = QLabel(self)
        self._unit_label = QLabel(self)
        self._cache = QSpinBox(self)

        self._cache.setMinimum(min_value)
        self._cache.setMaximum(max_value)
        self._cache.setValue(cache)
        self._cache.setSingleStep(inc)
        self._enabled_checkbox.setChecked(enabled)
        self._label.setText('Cache size: ')
        self._unit_label.setText(f'/{self._max_value} {self._unit}')
        self._cache.setDisabled(not enabled)

        # Signals
        self._enabled_checkbox.stateChanged.connect(self._on_enabled_checkbox)

        self._cache.valueChanged.connect(self._update_cache)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self._enabled_checkbox)
        layout.addWidget(self._label)
        layout.addWidget(self._cache)
        layout.addWidget(self._unit_label)

        self.setLayout(layout)

    def set_unit(self, value):
        """Update unit string displayed on widget

        Parameter
        ---------
        value: str
            New unit string.
        """

        self._unit = value
        self._update_unit_label()

    def _update_cache(self, value):
        '''Update dask cache value and emits signal that value was changed.

        Parameters
        ----------
        value: int
            New dask cache value in mb.

        '''

        if value > self._max_value:
            self._cache_value = self._max_value
        elif value < self._min_value:
            self._cache_value = self._min_value
        else:
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

    def _update_unit_label(self):
        '''Update string displayed next to cache value.'''
        new_string = f'/{self._max_value} {self._unit}'
        self._unit_label.setText(new_string)

    def setDescription(self, value):
        '''Set description of dask settings widget.

        Parameters
        ----------
        value: str
            Description for dask settings widget.
        '''
        self._description = value

    def cacheValue(self):
        """Return cache value."""

        return self._cache_value

    def value(self):
        """Return current dask cache value.

        Returns
        -------
        value: dict
            Current value of dask widget.
            {'enabled': self._enabled, 'cache': self._cache_value}
            enabled: bool
            cache: int (mb)
        """
        value = {
            'enabled': self._enabled,
            'cache': self._cache_value,
        }
        return value

    def setValue(self, value):
        """Set new value and update widget.

        Parameters
        ----------
        value : dict
            Dask cache value.
            value = {'enabled': self._enabled, 'cache': self._cache_value}
            enabled: bool
            cache: int (mb)
        """

        if value == "":
            return

        if value == {}:
            return

        cache = int(value['cache'])
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
            Minimum dask cache value in mb.
        """
        value = int(value)
        if value < self._max_value:
            self._min_value = value
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
            Minimum value of cache widget in mb.
        """
        return self._min_value

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

    def increment(self):
        """Return increment in dask cache value spinbox.

        Returns
        -------
        int
            Increment value of dask cache in mb.
        """
        return self._increment

    def setMaximum(self, value):
        """Set maximum dask cache value.

        Parameters
        ----------
        value : int
            Maximum dask cache value in mb.
        """
        value = int(value)
        if value > self._min_value:
            self._max_value = value
            self._cache_value = (
                self._max_value
                if self._cache_value > self._max_value
                else self._cache_value
            )
            self._cache.setMaximum(value)
            self._update_unit_label()
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
            Maximum value of dask cache in mb.
        """
        return self._max_value
