from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all(
    'skimage.io._plugins', include_py_files=True
)
datas = [f for f in datas if not f[0].endswith('.pyc')]
