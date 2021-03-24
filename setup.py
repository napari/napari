"""
PEP 517 doesn’t support editable installs
so this file is currently here to support "pip install -e ."
"""
import subprocess
import sys
from distutils.command.build_py import build_py as _build_py

from setuptools import setup


class build_py(_build_py):
    def run(self):
        # build type stubs
        python = sys.executable
        subprocess.run([python, '-m', 'napari.view_layers'])
        subprocess.run([python, '-m', 'napari.components.viewer_model'])
        super().run()


setup(
    use_scm_version={"write_to": "napari/_version.py"},
    setup_requires=["setuptools_scm"],
    cmdclass={"build_py": build_py},
)
