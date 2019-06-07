#!/usr/bin/env python
"""Napari GUI

GUI component of Napari.
"""

MIN_PY_VER = '3.6'
DISTNAME = 'napari'
DESCRIPTION = 'n-dimensional array viewer in Python.'
LONG_DESCRIPTION = __doc__
LICENSE = 'BSD 3-Clause'
DOWNLOAD_URL = 'https://github.com/napari/napari'

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Environment :: X11 Applications :: Qt',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
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


import os
import os.path as osp
import sys
from setuptools import setup, find_packages

import versioneer


if sys.version_info < (3, 6):
    sys.stderr.write(
        f'You are using Python '
        + "{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        + 'napari only supports Python 3.6 and above.\n\n'
        + 'Please install Python 3.6 using:\n'
        + '  $ pip install python==3.6\n\n'
    )
    sys.exit(1)


PACKAGES = [
    package for package in find_packages() if not package.startswith('gui')
]


with open(osp.join('requirements', 'default.txt')) as f:
    requirements = [
        line.strip() for line in f if line and not line.startswith('#')
    ]


INSTALL_REQUIRES = []
REQUIRES = []

for l in requirements:
    sep = l.split(' #')
    INSTALL_REQUIRES.append(sep[0].strip())
    if len(sep) == 2:
        REQUIRES.append(sep[1].strip())


if __name__ == '__main__':
    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        python_requires=f'>={MIN_PY_VER}',
        packages=PACKAGES,
        include_package_data=True,
        zip_safe=False,  # the package can run out of an .egg file
    )
