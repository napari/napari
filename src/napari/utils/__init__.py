from lazy_loader import attach as _attach

_proto_all_ = [
    '',
    'colormaps',
    'events',
    'settings',
    'transforms',
    'color',
    'io',
    'logo',
]
_submod_attrs = {
    '_check_numpy_version': ['NUMPY_VERSION_IS_THREADSAFE'],
    'colormaps.colormap': [
        'Colormap',
        'CyclicLabelColormap',
        'DirectLabelColormap',
    ],
    'notebook_display': ['NotebookScreenshot', 'nbscreenshot'],
    'progress': ['cancelable_progress', 'progrange', 'progress'],
    'info': ['citation_text', 'sys_info'],
    '_dask_utils': ['resize_dask_cache'],
}

__getattr__, __dir__, __all__ = _attach(
    __name__, submodules=_proto_all_, submod_attrs=_submod_attrs
)

del _attach
