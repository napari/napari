#!/bin/sh

# apt-get install libzmq3-dev

echo "removing old files..."

rm -rf build
rm -rf dist

echo "installing build tools..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install pyinstaller==3.6;
else
    # 3.5 seems to be required for PySide2 on linux, otherwise, in 3.4
    # it fails to find python in run-time hook 'pyi_rth_qt5plugins.py'
    pip install pyinstaller==3.5;
fi

conda install -y freetype


echo "building app..."


# find the directory of this script (for easier portability)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )";
else
    DIR=$(dirname $(readlink -f $0));
fi

python -m napari.resources._pkg_tools

if [ -f "$DIR/napari.spec" ]; then
    pyinstaller --noconfirm --clean --log-level=INFO "$DIR/napari.spec";
else
    echo "Could not find $DIR/napari.spec... quitting";
    exit 127;
fi


# pyqt5 works out of the box, but with PySide2, you may get the following error
# with pyinstaller 3.6, when running the executable:
# WARNING: Could not find the Qt platform plugin "___" in ""
# WARNING: This application failed to start because no Qt platform plugin
#   could be initialized. Reinstalling the application may fix this problem.
# not sure if this is a pyinstaller issue, but this fixes it
if [ -d "dist/napari.app/Contents/MacOS/PySide2" ]; then
    export SRC_DIR=$(pwd)
    for f in "$SRC_DIR"/dist/napari.app/Contents/MacOS/PySide2/plugins/*
    do
    ln -s "$f" "$SRC_DIR/dist/napari.app/Contents/MacOS/"
    done
fi


if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "creating mac .dmg ..."
    mkdir dist/dmg
    ln -s /Applications dist/dmg
    cp -r dist/napari.app dist/dmg
    hdiutil create dist/napari.dmg -srcfolder dist/dmg
    rm -rf dist/dmg
    rm -rf dist/napari

    # broken pkg building command
    # productbuild --component ./dist/napari.app /Applications napari.pkg \
    #   --sign "3rd Party Mac Developer Installer: napari team"
fi
