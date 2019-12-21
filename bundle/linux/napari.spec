# -*- mode: python ; coding: utf-8 -*-

import sys
from os.path import join, abspath
from napari import __version__
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE

sys.modules['FixTk'] = None

block_cipher = None

NAPARI_BASE = join("../..", "napari")
data_files = [
    (
        join(NAPARI_BASE, "utils", "colormaps", "matplotlib_cmaps.txt"),
        join("utils", "colormaps"),
    ),
    (join(NAPARI_BASE, "resources", "stylesheet.qss"), join("resources"),),
]

a = Analysis(
    ['../../napari/__main__.py'],
    pathex=[abspath('..')],
    binaries=[],
    datas=data_files,
    hiddenimports=[],
    hookspath=['../hooks'],
    runtime_hooks=[],
    excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='napari',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='napari',
)

app = BUNDLE(
    coll,
    name='napari.app',
    icon='logo.icns',
    bundle_identifier='com.napari.napari',
    info_plist={
        'CFBundleIdentifier': 'com.napari.napari',
        'CFBundleShortVersionString': __version__,
        'NSHighResolutionCapable': 'True',
    },
)
