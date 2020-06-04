import configparser
import os
import re
import shutil
import subprocess
import sys
import time

import tomlkit

WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")
HERE = os.path.abspath(os.path.dirname(__file__))
PYPROJECT_TOML = os.path.join(HERE, 'pyproject.toml')
SETUP_CFG = os.path.join(HERE, 'setup.cfg')

if WINDOWS:
    BUILD_DIR = os.path.join(HERE, 'windows')
elif LINUX:
    BUILD_DIR = os.path.join(HERE, 'linux')
elif MACOS:
    BUILD_DIR = os.path.join(HERE, 'macOS')


with open(PYPROJECT_TOML, 'r') as f:
    original_toml = f.read()

with open(os.path.join(HERE, "napari", "_version.py")) as f:
    match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
    if match:
        VERSION = match.groups()[0].split('.dev')[0]


def patch_toml():
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG)
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = [r.split('#')[0].strip() for r in requirements if r]

    toml = tomlkit.parse(original_toml)
    toml['tool']['briefcase']['version'] = VERSION
    toml['tool']['briefcase']['app']['napari']['requires'] = requirements + [
        "pip",
        "PySide2==5.14.2.2",
    ]

    print("patching pyroject.toml to version: ", VERSION)
    print(
        "patching pyroject.toml requirements to : \n",
        "\n".join(toml['tool']['briefcase']['app']['napari']['requires']),
    )

    with open(PYPROJECT_TOML, 'w') as f:
        f.write(tomlkit.dumps(toml))


def patch_dmgbuild():
    if not MACOS:
        return
    from dmgbuild import core

    # will not be required after dmgbuild > v1.3.3
    # see https://github.com/al45tair/dmgbuild/pull/18
    with open(core.__file__, 'r') as f:
        src = f.read()
    if 'max(total_size / 1024' not in src:
        return
    with open(core.__file__, 'w') as f:
        f.write(src.replace('max(total_size / 1024', 'max(total_size / 1000'))
        print("patched dmgbuild.core")


def patch_wxs():
    # must run after briefcase create
    fname = os.path.join(BUILD_DIR, 'napari', 'napari.wxs')

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            source = f.read()
        with open(fname, 'w') as f:
            f.write(source.replace('pythonw.exe', 'python.exe'))
            print("patched pythonw.exe -> python.exe")


def make_zip():
    import zipfile
    import glob

    if WINDOWS:
        ext, OS = '*.msi', 'Windows'
    elif LINUX:
        ext, OS = '*.AppImage', 'Linux'
    elif MACOS:
        ext, OS = '*.dmg', 'macOS'
    artifact = glob.glob(os.path.join(BUILD_DIR, ext))[0]
    dest = f'napari-{VERSION}-{OS}.zip'

    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact, arcname=os.path.basename(artifact))
    print("created zipfile: ", dest)

    return dest


def clean():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def bundle():
    clean()

    if MACOS:
        patch_dmgbuild()

    # smoke test, and build resources
    subprocess.check_call([sys.executable, '-m', 'napari', '--info'])
    patch_toml()

    # create
    subprocess.check_call(['briefcase', 'create'])
    time.sleep(0.5)

    if WINDOWS:
        patch_wxs()

    # build
    subprocess.check_call(['briefcase', 'build'])

    # package
    cmd = ['briefcase', 'package'] + (['--no-sign'] if MACOS else [])
    subprocess.check_call(cmd)

    # compress
    dest = make_zip()
    clean()

    with open(PYPROJECT_TOML, 'w') as f:
        f.write(original_toml)

    return dest


if __name__ == "__main__":
    if '--clean' in sys.argv:
        clean()
        sys.exit()
    print('created', bundle())
