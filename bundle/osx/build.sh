#!/bin/sh


#get the version string
version=`python3 -c "import os, sys;tmp = sys.stdout;sys.stdout = open(os.devnull,'w');sys.stderr= open(os.devnull,'w');import spimagine;sys.stdout = tmp;print(spimagine.__version__)"`


echo "removing old files..."

rm -rf build
rm -rf dist


echo "checking installations of build tools"

pip install pyinstaller==3.5
conda install freetype


echo "building app..."

pyinstaller --windowed --onefile --noconfirm --clean --log-level=INFO napari.spec
