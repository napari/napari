import platform

import numpy as np
from packaging.version import parse as parse_version

# if run with numpy<2 on macOS arm64 architecture compiled from pypi wheels,
# then it will crash with bus error if numpy is used in different thread
# Issue reported https://github.com/numpy/numpy/issues/21799
if (
    parse_version(np.__version__) < parse_version('2')
    and platform.system() == 'Darwin'
    and platform.machine() == 'arm64'
):  # pragma: no cover
    try:
        PROBLEMATIC_NUMPY_MACOS = (
            'cibw-run' in np.show_config('dicts')['Python Information']['path']
        )
    except (KeyError, TypeError):
        PROBLEMATIC_NUMPY_MACOS = False
else:
    PROBLEMATIC_NUMPY_MACOS = False
