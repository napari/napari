from napari.utils.compat import StrEnum


class ColormapBackend(StrEnum):
    """
    Enum for colormap backends.

    Attributes
    ----------
    numba : ColormapBackend
        Use Numba for colormap operations.
    pure_python : ColormapBackend
        Use pure Python for colormap operations.
    fastest_available : ColormapBackend
        Use the fastest available backend, which may be Numba or a compiled backend.
    """

    fastest_available = 'Fastest available'
    pure_python = 'Pure Python'
    numba = 'numba'
    partsegcore = 'PartSegCore'

    def __str__(self) -> str:
        """Return the string representation of the backend."""
        return str(self.value)

    def __repr__(self) -> str:
        """Return the string representation of the backend."""
        return self.name

    @classmethod
    def _missing_(cls, value: object) -> 'ColormapBackend':
        """Handle missing values in the enum."""
        # Handle the case where the value is not a valid enum member
        if isinstance(value, str):
            return cls[value.replace(' ', '_').lower()]
        raise ValueError(f"'{value}' is not a valid ColormapBackend.")


def get_backend() -> ColormapBackend:
    from napari.utils.colormaps import _accelerated_cmap

    return _accelerated_cmap.COLORMAP_BACKEND


def set_backend(backend: ColormapBackend) -> ColormapBackend:
    """Set the colormap backend to use.

    Parameters
    ----------
    backend : ColormapBackend
        The colormap backend to use.

    Returns
    -------
    ColormapBackend
        The previous colormap backend.
    """
    from napari.utils.colormaps import _accelerated_cmap

    previous_backend = _accelerated_cmap.COLORMAP_BACKEND
    _accelerated_cmap.set_colormap_backend(backend)

    return previous_backend
