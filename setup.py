#!/usr/bin/env python
"""Napari viewer is a fast, interactive, multi-dimensional image viewer for
Python. It's designed for browsing, annotating, and analyzing large
multi-dimensional images. It's built on top of `Qt` (for the GUI), `vispy`
(for performant GPU-based rendering), and the scientific Python stack
(`numpy`, `scipy`).
"""

import os.path as osp
import sys
from setuptools import find_packages, setup

import versioneer

MIN_PY_MAJOR_VER = 3
MIN_PY_MINOR_VER = 6
MIN_PY_VER = f"{MIN_PY_MAJOR_VER}.{MIN_PY_MINOR_VER}"
DISTNAME = 'napari'
DESCRIPTION = 'n-dimensional array viewer in Python.'
LONG_DESCRIPTION = __doc__
LICENSE = 'BSD 3-Clause'
DOWNLOAD_URL = 'https://github.com/napari/napari'

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: X11 Applications :: Qt',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Utilities',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]


if sys.version_info < (MIN_PY_MAJOR_VER, MIN_PY_MINOR_VER):
    sys.stderr.write(
        f"You are using Python "
        f"{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        f"napari only supports Python {MIN_PY_VER} and above.\n\n"
        f"Please install Python {MIN_PY_VER} or later.\n"
    )
    sys.exit(1)

requirements = []
with open(osp.join('requirements', 'default.txt')) as f:
    for line in f:
        splitted = line.split("#")
        stripped = splitted[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    download_url=DOWNLOAD_URL,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    classifiers=CLASSIFIERS,
    install_requires=requirements,
    python_requires=f'>={MIN_PY_VER}',
    packages=find_packages(),
    entry_points={'console_scripts': ['napari=napari.__main__:main']},
    include_package_data=True,
    zip_safe=False,  # the package can run out of an .egg file
)
