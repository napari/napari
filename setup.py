#!/usr/bin/env python
"""Napari GUI

GUI component of Napari.
"""

MIN_PY_VER = '3.6'
DISTNAME = 'napari-gui'
DESCRIPTION = 'GUI component of Napari.'
LONG_DESCRIPTION = __doc__
LICENSE = 'BSD 3-Clause'
PACKAGES = ['napari_gui']
DOWNLOAD_URL = 'https://github.com/Napari/napari-gui'

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
    'Operating System :: MacOS'
]


import os
import sys
from setuptools import setup
import versioneer

if sys.version_info < (3, 6):
    sys.stderr.write(f'You are using Python '
                     + "{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
                     + 'napari only supports Python 3.6 and above.\n\n'
                     + 'Please install Python 3.6 using:\n'
                     + '  $ pip install python==3.6\n\n')
    sys.exit(1)

with open('requirements.txt') as f:
    requirements = [l.strip() for l in f.readlines() if l]


INSTALL_REQUIRES = []
REQUIRES = []

for l in requirements:
    if l.startswith('#'):  # it's a comment
        requirements.remove(l)

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
        zip_safe=False  # the package can run out of an .egg file
    )
