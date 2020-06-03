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


def patch_toml():
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG)
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = [r.split('#')[0].strip() for r in requirements if r]

    with open(PYPROJECT_TOML, 'r') as f:
        toml = tomlkit.parse(f.read())

    with open(os.path.join(HERE, "napari", "_version.py")) as f:
        match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
        if match:
            version = match.groups()[0].split('.dev')[0]
        else:
            version = ''

    toml['tool']['briefcase']['version'] = version
    toml['tool']['briefcase']['app']['napari']['requires'] = requirements + [
        "pip",
        "PySide2==5.14.2.2",
    ]

    print("patching pyroject.toml to version: ", version)
    print(
        "patching pyroject.toml requirements to : \n",
        "\n".join(toml['tool']['briefcase']['app']['napari']['requires']),
    )

    with open(PYPROJECT_TOML, 'w') as f:
        f.write(tomlkit.dumps(toml))


def clean():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def create():
    clean()
    subprocess.check_call(['briefcase', 'create'])
    time.sleep(0.5)


def build():
    subprocess.check_call(['briefcase', 'build'])


def package():
    subprocess.check_call(['briefcase', 'package'])


def import_once():
    if WINDOWS:
        binary = os.path.join(
            BUILD_DIR, 'napari', 'src', 'python', 'python.exe'
        )

    subprocess.check_call([binary, '-m', 'napari', '--info'])


def bundle():
    patch_toml()
    create()
    import_once()  # smoke test, and build resources
    build()
    package()


if __name__ == "__main__":
    if '--clean' in sys.argv:
        sys.exit(clean())
    bundle()
