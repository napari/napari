# -*- mode: python ; coding: utf-8 -*-

import sys
from os.path import abspath, join
from napari import __version__
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE
from PyInstaller.utils.hooks import collect_data_files

sys.modules['FixTk'] = None


def keep(x):
    if any(x.endswith(e) for e in ('.svg', '.DS_Store', '.qrc')):
        return False
    return True


def format(x):
    base = join("..", "..", "napari")
    x0 = join(base, f"{x[0].split('napari/')[-1]}")
    x1 = f"{x[1].split('napari/')[-1]}"
    return (x0, x1)


DATA_FILES = [format(f) for f in collect_data_files('napari') if keep(f[0])]
BLOCK_CIPHER = None
NAME = 'napari'

a = Analysis(
    ['../../napari/__main__.py'],
    pathex=[abspath('..')],
    binaries=[],
    datas=DATA_FILES,
    hiddenimports=[],
    hookspath=['../hooks'],
    runtime_hooks=[],
    excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=BLOCK_CIPHER)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
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
    name=NAME,
)

app = BUNDLE(
    coll,
    name=NAME + '.app',
    icon='logo.icns',
    bundle_identifier=f'com.{NAME}.{NAME}',
    info_plist={
        'CFBundleIdentifier': f'com.{NAME}.{NAME}',
        'CFBundleShortVersionString': __version__,
        'NSHighResolutionCapable': 'True',
    },
)
