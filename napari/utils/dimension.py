"""Scale bar dimension"""
import bisect
from operator import itemgetter
from typing import Optional

PREFIXES_FACTORS = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "\u00b5": 1e-6,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
    "z": 1e-21,
    "y": 1e-24,
}


PREFERRED_VALUES = [
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    50,
    75,
    100,
    125,
    150,
    200,
    500,
    750,
]


class Dimension(object):
    """Base dimension class"""

    def __init__(self, base_unit: str, current_unit: Optional[str] = None):
        self._units = {base_unit: 1.0}
        self._base_unit = base_unit
        self.current_unit = (
            current_unit if current_unit is not None else base_unit
        )

    @classmethod
    def from_unit(cls, unit: str):
        """Create instance of the class based on the units"""
        return cls(unit)

    @classmethod
    def from_factor(cls, factor: float):
        """Create instance of the class based on the provided factor"""
        _cls = cls()  # noqa, this is expected to be used by the subclasses
        unit = None
        for unit, _factor in _cls._units.items():
            if _factor == factor:
                break
        if unit is None:
            raise ValueError(
                "Could not find appropriate unit for specified factor"
            )
        return cls.from_unit(unit)

    @property
    def units(self):
        """List of units"""
        return list(self._units.keys())

    @property
    def current_unit(self) -> str:
        """Get current unit"""
        return self._current_unit

    @current_unit.setter
    def current_unit(self, unit: Optional[str] = None):
        """Set current unit"""
        if unit is None:
            unit = self._base_unit
        if unit not in self._units:
            raise ValueError(
                f"Could not find unit `{unit}` in the units dictionary. Try:\n{','.join(self._units.keys())}"
            )
        self._current_unit = unit

    def add_units(self, units: str, factor: float):
        """Add new possible units.

        Parameters
        ----------
        units : str
            units
        factor : float
            multiplication factor to convert new units into base units
        """
        if units in self._units:
            raise ValueError("%s already defined" % units)
        if factor == 1:
            raise ValueError("Factor cannot be equal to 1")
        self._units[units] = factor

    def calculate_preferred(self, value: float):
        """Calculate value and units

        Parameters
        ----------
        value : float
            value to check against the predefined units and factors

        Returns
        -------
        new_value : float
            value after it was converted
        unit : str
            new units
        """
        unit = self.current_unit
        # if units not in self._units:
        #     raise ValueError("Unknown units: %s" % units)
        base_value = value * self._units[unit]

        units_factor = sorted(self._units.items(), key=itemgetter(1))
        factors = [item[1] for item in units_factor]
        index = bisect.bisect_right(factors, base_value)

        if index:
            new_units, factor = units_factor[index - 1]
            return base_value / factor, new_units
        return value, unit


class NullDimension(Dimension):
    """Null dimension where no units are provided"""

    def __init__(self, current_unit: Optional[str] = None):
        super().__init__("")
        self.current_unit = current_unit


class SILengthDimension(Dimension):
    """SI dimension"""

    def __init__(self, current_unit: Optional[str] = None):
        super().__init__("m")
        for prefix, factor in PREFIXES_FACTORS.items():
            self.add_units(prefix + "m", factor)
        self.current_unit = current_unit


class SILengthReciprocalDimension(Dimension):
    """Reciprocal SI dimension"""

    def __init__(self, current_unit: Optional[str] = None):
        super().__init__("1/m")
        for prefix, factor in PREFIXES_FACTORS.items():
            self.add_units(f"1/{prefix}m", 1 / factor)
        self.current_unit = current_unit


class ImperialLengthDimension(Dimension):
    """Imperial units dimension"""

    def __init__(self, current_unit: Optional[str] = None):
        super().__init__("ft")
        self.add_units("th", 1 / 12000)
        self.add_units("in", 1 / 12)
        self.add_units("yd", 3)
        self.add_units("ch", 66)
        self.add_units("fur", 660)
        self.add_units("mi", 5280)
        self.add_units("lea", 15840)
        self.current_unit = current_unit


class PixelLengthDimension(Dimension):
    """Pixel dimension"""

    def __init__(self, current_unit: Optional[str] = None):
        super().__init__("px")
        for prefix, factor in PREFIXES_FACTORS.items():
            if factor < 1:
                continue
            self.add_units(prefix + "px", factor)
        self.current_unit = current_unit


# Dictionary of registered Dimensions : units
DIMENSIONS = {
    PixelLengthDimension: PixelLengthDimension().units,
    SILengthDimension: SILengthDimension().units,
    SILengthReciprocalDimension: SILengthReciprocalDimension().units,
    ImperialLengthDimension: ImperialLengthDimension().units,
}
# list of units where pixel size should be 1
ONE_PIXEL_SIZE = [""] + PixelLengthDimension().units
