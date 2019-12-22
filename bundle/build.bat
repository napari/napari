echo "removing old files..."

rmdir /s /q build
rmdir /s /q dist

echo "checking installations of build tools"

REM with pyinstaller 3.5 and pyside2 I get the following error
REM https://stackoverflow.com/questions/57932432/pyinstaller-win32ctypes-pywin32-pywintypes-error-2-loadlibraryexw-the-sys
rem pip install pyinstaller==3.4
rem conda install -y freetype

echo "building app..."


pyinstaller --noconfirm --clean --log-level=INFO napari.spec
