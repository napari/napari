from PyInstaller.utils.hooks import collect_all


datas, binaries, hiddenimports = collect_all('pip', include_py_files=False)

hiddenimports += ['setuptools', 'pip._internal.resolution']
