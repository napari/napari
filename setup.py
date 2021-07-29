"""
PEP 517 doesn’t support editable installs
so this file is currently here to support "pip install -e ."
"""
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


def _build_stubs():
    subprocess.run([sys.executable, '-m', 'napari.utils.stubgen'])


class build_py(_build_py):
    def run(self) -> None:
        _build_stubs()
        return super().run()


class sdist(_sdist):
    def run(self) -> None:
        _build_stubs()
        return super().run()


setup(
    use_scm_version={"write_to": "napari/_version.py"},
    setup_requires=["setuptools_scm"],
    cmdclass={"build_py": build_py, 'sdist': sdist},
)
