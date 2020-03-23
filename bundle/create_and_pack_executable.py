import codecs
import os
import platform
import re
import shutil
import sys
import tarfile
import zipfile

import napari

from PyInstaller.__main__ import run as pyinstaller_run

if len(sys.argv) == 2:
    base_path = os.path.abspath(sys.argv[1])
else:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["BUNDLE_ROOT"] = base_dir

pyinstaller_run(["-y", "--debug=all", os.path.join(base_dir, "napari.spec")])


def read(*parts):
    with codecs.open(os.path.join(base_path, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = napari.__version__

name_dict = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}

system_name = name_dict[platform.system()]

os.makedirs(os.path.join(base_path, "dist2"), exist_ok=True)


pyside_path = os.path.join(os.curdir, "dist", "napari", "PySide2")
if os.path.exists(os.path.join(pyside_path, "plugins")):
    shutil.move(
        os.path.join(pyside_path, "plugins"), os.path.join(pyside_path, "Qt")
    )


if platform.system() == "Darwin":
    arch_file = tarfile.open(
        os.path.join(
            base_path, "dist2", f"napari-{version}-{system_name}.tgz"
        ),
        'w:gz',
    )
    arch_file.write = arch_file.add
else:
    arch_file = zipfile.ZipFile(
        os.path.join(
            base_path, "dist2", f"napari-{version}-{system_name}.zip"
        ),
        'w',
        zipfile.ZIP_DEFLATED,
    )

base_zip_path = os.path.join(os.curdir, "dist")

if platform.system() == "Darwin":
    dir_name = "napari.app"
else:
    dir_name = "napari"

for root, dirs, files in os.walk(
    os.path.join(base_zip_path, dir_name), topdown=False, followlinks=True
):
    for file_name in files:
        arch_file.write(
            os.path.join(root, file_name),
            os.path.relpath(os.path.join(root, file_name), base_zip_path),
        )

arch_file.close()
