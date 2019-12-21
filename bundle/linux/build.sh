#!/bin/sh


#get the version string
version=`python3 -c "import os, sys;tmp = sys.stdout;sys.stdout = open(os.devnull,'w');sys.stderr= open(os.devnull,'w');import spimagine;sys.stdout = tmp;print(spimagine.__version__)"`

# apt-get install libzmq3-dev

echo "removing old files..."

rm -rf build
rm -rf dist


echo "checking installations of build tools"

pip install pyinstaller==3.5
conda install freetype


echo "building app..."

pyinstaller --windowed --onefile --noconfirm --clean --log-level=INFO napari.spec

# pyqt5 works out of the box, but with PySide2, you may get the following error
# WARNING: Could not find the Qt platform plugin "xcb" in ""
# WARNING: This application failed to start because no Qt platform plugin
#   could be initialized. Reinstalling the application may fix this problem.
# not sure if this is a pyinstaller issue, but this fixes it
if [ -d "dist/napari/PySide2/plugins" ]; then
    mv "dist/napari/PySide2/plugins/" "dist/napari/PySide2/Qt/";
fi
