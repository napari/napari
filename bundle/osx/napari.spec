# -*- mode: python ; coding: utf-8 -*-

import sys
import vispy.glsl
import vispy.io
import freetype
sys.modules['FixTk'] = None

block_cipher = None

data_files = [
    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data")),
    (os.path.dirname(vispy.util.__file__), os.path.join("vispy", "util")),
    (os.path.dirname(freetype.__file__), os.path.join("freetype")),
    (os.path.join("../..", "napari", "util", "colormaps", "matplotlib_cmaps.txt"),
     os.path.join("util", "colormaps")),
    (os.path.join("../..", "napari", "resources", "stylesheet.qss"), "resources"),
]

hidden_imports = [
    "vispy.ext._bundled.six",
    "vispy.app.backends._pyside2",
    "freetype"
]

a = Analysis(['../../napari/main.py'],
             pathex=['/Users/nicholassofroniew/Github/napari'],
             binaries=[],
             datas=data_files,
             hiddenimports=hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='napari',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='napari')
app = BUNDLE(coll,
             name='napari.app',
             icon=None,
             bundle_identifier=None)
