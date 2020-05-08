#!/usr/bin/env python
"""Napari viewer is a fast, interactive, multi-dimensional image viewer for
Python. It's designed for browsing, annotating, and analyzing large
multi-dimensional images. It's built on top of `Qt` (for the GUI), `vispy`
(for performant GPU-based rendering), and the scientific Python stack
(`numpy`, `scipy`).
"""

import os.path as osp
import sys
from setuptools import setup

import versioneer

MIN_PY_MAJOR_VER = 3
MIN_PY_MINOR_VER = 6
MIN_PY_VER = f"{MIN_PY_MAJOR_VER}.{MIN_PY_MINOR_VER}"
LONG_DESCRIPTION = __doc__
DOWNLOAD_URL = 'https://github.com/napari/napari'


if sys.version_info < (MIN_PY_MAJOR_VER, MIN_PY_MINOR_VER):
    sys.stderr.write(
        f"You are using Python "
        f"{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        f"napari only supports Python {MIN_PY_VER} and above.\n\n"
        f"Please install Python {MIN_PY_VER} or later.\n"
    )
    sys.exit(1)

QT_MIN_VERSION = "5.12.3"

try:
    import PyQt5  # noqa: F401

    pyqt = True
except ImportError:
    pyqt = False

try:
    import PySide2  # noqa: F401

    pyside = True
except ImportError:
    pyside = False

requirements = []
with open(osp.join('requirements', 'default.txt')) as f:
    for line in f:
        splitted = line.split("#")
        stripped = splitted[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

if pyqt:
    requirements.append("PyQt5>=" + QT_MIN_VERSION)
if pyside or not (pyqt or pyside):
    requirements.append("PySide2>=" + QT_MIN_VERSION)


setup(
    long_description=LONG_DESCRIPTION,
    download_url=DOWNLOAD_URL,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=requirements,
)
