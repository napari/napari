from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('vispy')

hiddenimports = [
    "vispy.ext._bundled.six",
    "vispy.app.backends._pyqt5",
    "vispy.app.backends._pyside2",
]
