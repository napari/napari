import sys
import os


if sys.platform == 'darwin':
    from dmgbuild import core
    from briefcase.platforms.macOS import app

    with open(core.__file__, 'r') as f:
        source = f.read()
    source = source.replace('max(total_size / 1024', 'max(total_size / 1000')
    with open(core.__file__, 'w') as f:
        f.write(source)
        print("patched dmgbuild.core")

    with open(app.__file__, 'r') as f:
        source = f.read()
    source = source.replace(
        "# Obtain the valid codesigning identities.",
        'if identity == "-": return "-"',
    )
    with open(app.__file__, 'w') as f:
        f.write(source)
        print("patched briefcase.platforms.macOS.app")


if sys.platform.startswith("win"):
    # must run after briefcase build
    fname = 'windows\\napari\\napari.wxs'
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            source = f.read()
        with open(fname, 'w') as f:
            f.write(source.replace('pythonw.exe', 'python.exe'))
            print("patched pythonw.exe -> python.exe")
