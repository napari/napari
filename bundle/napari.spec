# -*- mode: python ; coding: utf-8 -*-

import sys
from os.path import abspath, join, dirname, pardir
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE
from PyInstaller.utils.hooks import collect_data_files
import napari

sys.modules['FixTk'] = None

NAPARI_ROOT = dirname(napari.__file__)
BUNDLE_ROOT = abspath(join(NAPARI_ROOT, pardir, 'bundle'))


def get_icon():
    logo_file = 'logo.ico' if sys.platform.startswith('win') else 'logo.icns'
    return join(BUNDLE_ROOT, logo_file)


def get_version():
    if sys.platform != 'win32':
        return None
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo,
        FixedFileInfo,
        StringFileInfo,
        StringTable,
        StringStruct,
        VarFileInfo,
        VarStruct,
    )

    ver_str = napari.__version__
    version = ver_str.replace("+", '.').split('.')
    version = [int(p) for p in version if p.isnumeric()]
    version += [0] * (4 - len(version))
    # The following is a hack specific to pyinstaller 3.5 that is needed to
    # pass a VSVersionInfo directly to EXE(): EXE assumes that the `version`
    # argument is a path-like object and therefore has to be tricked into
    # ignoring it by exhibiting a falsy value in boolean context. However, the
    # object is later passed into the `SetVersion` function which can also
    # handle VSVersionInfo directly.
    # class VersionInfo(VSVersionInfo):
    #     _count = 0

    #     def __bool__(self):
    #         self._count += 1
    #         return self._count > 1

    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=tuple(version)[:4], prodvers=tuple(version)[:4],
        ),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        '040904E4',
                        [
                            StringStruct('CompanyName', 'napari'),
                            StringStruct('FileDescription', 'napari'),
                            StringStruct('FileVersion', ver_str),
                            StringStruct('InternalName', 'napari'),
                            StringStruct(
                                'LegalCopyright',
                                '© napari. All rights reserved.',
                            ),
                            StringStruct('OriginalFilename', 'napari.exe'),
                            StringStruct('ProductName', 'napari'),
                            StringStruct('ProductVersion', ver_str),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct(u'Translation', [0x409, 1252])]),
        ],
    )


def keep(x):
    if any(x.endswith(e) for e in ('.svg', '.DS_Store', '.qrc')):
        return False
    return True


def format(x):
    if sys.platform.startswith('win'):
        x0 = join(NAPARI_ROOT, x[0].split('napari\\')[-1])
        x1 = x[1].split('napari\\')[-1]
    else:
        x0 = join(NAPARI_ROOT, x[0].split('napari/')[-1])
        x1 = x[1].split('napari/')[-1]
    return (x0, x1)


DATA_FILES = [format(f) for f in collect_data_files('napari') if keep(f[0])]
NAME = 'napari'
WINDOWED = True
DEBUG = False
UPX = False
BLOCK_CIPHER = None
HOOKSPATH = join(BUNDLE_ROOT, 'hooks')
print("HOOKS PATH:", HOOKSPATH)

a = Analysis(
    [join(NAPARI_ROOT, '__main__.py')],
    pathex=[BUNDLE_ROOT],
    datas=DATA_FILES,
    hookspath=[HOOKSPATH],
    excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
    cipher=BLOCK_CIPHER,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=BLOCK_CIPHER)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=DEBUG,
    upx=UPX,
    console=(not WINDOWED),
    icon=get_icon(),
    version=get_version(),
)

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, upx=UPX, name=NAME,)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=NAME + '.app',
        icon=get_icon(),
        bundle_identifier=f'com.{NAME}.{NAME}',
        info_plist={
            'CFBundleIdentifier': f'com.{NAME}.{NAME}',
            'CFBundleShortVersionString': napari.__version__,
            'NSHighResolutionCapable': 'True',
        },
    )
