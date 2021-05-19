"""
PEP 517 doesn’t support editable installs
so this file is currently here to support "pip install -e ."
"""
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self) -> None:
        modules = ['napari.view_layers', 'napari.components.viewer_model']
        cmd = [sys.executable, '-m', 'napari.utils.stubgen']
        subprocess.run(cmd + modules)
        return super().run()


setup(
    use_scm_version={"write_to": "napari/_version.py"},
    setup_requires=["setuptools_scm"],
    cmdclass={"build_py": build_py},
)
