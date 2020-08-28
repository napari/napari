import os
import re
import shutil
import subprocess
import sys
import time

import tomlkit

APP = 'napari'
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
    APP_DIR = os.path.join(BUILD_DIR, APP, f'{APP}.app')


with open(PYPROJECT_TOML, 'r') as f:
    original_toml = f.read()

with open(os.path.join(HERE, "napari", "_version.py")) as f:
    match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
    if match:
        VERSION = match.groups()[0].split('+')[0]


def update_toml_version():
    toml = tomlkit.parse(original_toml)
    toml['tool']['briefcase']['version'] = VERSION
    print("updating pyproject.toml to version: ", VERSION)
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


def add_site_packages_to_path():
    # on mac, make sure the site-packages folder exists even before the user
    # has pip installed, so it is in sys.path on the first run
    # (otherwise, newly installed plugins will not be detected until restart)
    if MACOS:
        pkgs_dir = os.path.join(
            APP_DIR,
            'Contents',
            'Resources',
            'Support',
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
        )
        os.makedirs(pkgs_dir)
        print("created site-packages at", pkgs_dir)

    # on windows, briefcase uses a _pth file to determine the sys.path at
    # runtime.  https://docs.python.org/3/using/windows.html#finding-modules
    # We update that file with the eventual location of pip site-packages
    elif WINDOWS:
        py = "".join(map(str, sys.version_info[:2]))
        python_dir = os.path.join(BUILD_DIR, APP, 'src', 'python')
        pth = os.path.join(python_dir, f'python{py}._pth')
        with open(pth, "a") as f:
            # Append 'hello' at the end of file
            f.write(".\\\\Lib\\\\site-packages\n")
        print("added bundled site-packages to", pth)

        pkgs_dir = os.path.join(python_dir, 'Lib', 'site-packages')
        os.makedirs(pkgs_dir)
        print("created site-packages at", pkgs_dir)
        with open(os.path.join(pkgs_dir, 'readme.txt'), 'w') as f:
            f.write("this is where plugin packages will go")


def patch_wxs():
    # must run after briefcase create
    fname = os.path.join(BUILD_DIR, APP, f'{APP}.wxs')

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            source = f.read()
        with open(fname, 'w') as f:
            f.write(source.replace('pythonw.exe', 'python.exe'))
            print("patched pythonw.exe -> python.exe")


def make_zip():
    import glob
    import zipfile

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
    subprocess.check_call([sys.executable, '-m', APP, '--info'])
    update_toml_version()

    # create
    cmd = ['briefcase', 'create'] + (['--no-docker'] if LINUX else [])
    subprocess.check_call(cmd)

    time.sleep(0.5)

    add_site_packages_to_path()

    if WINDOWS:
        patch_wxs()

    # build
    cmd = ['briefcase', 'build'] + (['--no-docker'] if LINUX else [])
    subprocess.check_call(cmd)

    # package
    cmd = ['briefcase', 'package']
    cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
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
    if '--version' in sys.argv:
        print(VERSION)
        sys.exit()
    print('created', bundle())
