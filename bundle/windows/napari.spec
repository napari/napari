# -*- mode: python ; coding: utf-8 -*-

import sys
from os.path import abspath, join, pardir, isfile
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_data_files
from napari import __version__

sys.modules['FixTk'] = None


def get_version():
    if sys.platform != 'win32':
        return None
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo,
        StringTable, StringStruct, VarFileInfo, VarStruct)

    ver_str = __version__
    version = ver_str.replace("+", '.').split('.')
    version = [int(p) for p in version if p.isnumeric()]
    version += [0] * (4 - len(version))
    # # The following is a hack specific to pyinstaller 3.5 that is needed to
    # # pass a VSVersionInfo directly to EXE(): EXE assumes that the `version`
    # # argument is a path-like object and therefore has to be tricked into
    # # ignoring it by exhibiting a falsy value in boolean context. However, the
    # # object is later passed into the `SetVersion` function which can also
    # # handle VSVersionInfo directly.
    # class VersionInfo(VSVersionInfo):
    #     _count = 0

    #     def __bool__(self):
    #         self._count += 1
    #         return self._count > 1

    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=tuple(version)[:4],
            prodvers=tuple(version)[:4],
        ),
        kids=[
            StringFileInfo([
                StringTable('040904E4', [
                    StringStruct('CompanyName', 'napari'),
                    StringStruct('FileDescription', 'napari'),
                    StringStruct('FileVersion', ver_str),
                    StringStruct('InternalName', 'napari'),
                    StringStruct('LegalCopyright', '© napari. All rights reserved.'),
                    StringStruct('OriginalFilename', 'napari.exe'),
                    StringStruct('ProductName', 'napari'),
                    StringStruct('ProductVersion', ver_str),
                ])
            ]),
            VarFileInfo([VarStruct(u'Translation', [0x409, 1252])])
        ]
    )


def keep(x):
    if any(x.endswith(e) for e in ('.svg', '.DS_Store', '.qrc')):
        return False
    return True


def format(x):
    base = join(pardir, pardir, "napari")
    x0 = join(base, x[0].split('napari\\')[-1])
    x1 = x[1].split('napari\\')[-1]
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
    upx=False,
    console=False,
    icon='logo.ico' if isfile('logo.ico') else None,
    version=get_version(),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NAME,
)