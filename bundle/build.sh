#!/bin/sh

# apt-get install libzmq3-dev

echo "removing old files..."

rm -rf build
rm -rf dist

echo "installing build tools..."

# 3.5 works fine for PyQt5... but lots of issues with PySide2
pip install pyinstaller==3.4
conda install -y freetype


echo "building app..."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pyinstaller --noconfirm --clean --log-level=INFO "$DIR/napari.spec"

# pyqt5 works out of the box, but with PySide2, you may get the following error
# with pyinstaller 3.5, when running the executable:
# WARNING: Could not find the Qt platform plugin "xcb" in ""
# WARNING: This application failed to start because no Qt platform plugin
#   could be initialized. Reinstalling the application may fix this problem.
# not sure if this is a pyinstaller issue, but this fixes it
if [ -d "dist/napari/PySide2/plugins" ]; then
    mv "dist/napari/PySide2/plugins/" "dist/napari/PySide2/Qt/";
fi
