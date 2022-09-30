import configparser
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import tomlkit

APP = 'napari'

# EXTRA_REQS will be added to the bundle, in addition to those specified in
# setup.cfg.  To add additional packages to the bundle, or to override any of
# the packages listed here or in `setup.cfg, use the `--add` command line
# argument with a series of "pip install" style strings when running this file.
# For example, the following will ADD ome-zarr, and CHANGE the version of
# PySide2:
# python bundle.py --add 'PySide2==5.15.0' 'ome-zarr'

# This is now defined in setup.cfg "options.extras_require.bundle_run"
# EXTRA_REQS = []

WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")
HERE = os.path.abspath(os.path.dirname(__file__))
PYPROJECT_TOML = os.path.join(HERE, 'pyproject.toml')
SETUP_CFG = os.path.join(HERE, 'setup.cfg')
ARCH = (platform.machine() or "generic").lower().replace("amd64", "x86_64")

if WINDOWS:
    BUILD_DIR = os.path.join(HERE, 'windows')
    APP_DIR = os.path.join(BUILD_DIR, APP, 'src')
    EXT, OS = 'msi', 'Windows'
elif LINUX:
    BUILD_DIR = os.path.join(HERE, 'linux')
    APP_DIR = os.path.join(BUILD_DIR, APP, f'{APP}.AppDir')
    EXT, OS = 'AppImage', 'Linux'
elif MACOS:
    BUILD_DIR = os.path.join(HERE, 'macOS')
    APP_DIR = os.path.join(BUILD_DIR, APP, f'{APP}.app')
    EXT, OS = 'dmg', 'macOS'

with open(os.path.join(HERE, "napari", "_version.py")) as f:
    match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
    if match:
        VERSION = match.groups()[0].split('+')[0]


@contextmanager
def patched_toml():
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG)
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = [r.split('#')[0].strip() for r in requirements if r]

    with open(PYPROJECT_TOML) as f:
        original_toml = f.read()

    toml = tomlkit.parse(original_toml)

    # Initialize EXTRA_REQS from setup.cfg 'options.extras_require.bundle_run'
    bundle_run = parser.get("options.extras_require", "bundle_run")
    EXTRA_REQS = [
        requirement.split('#')[0].strip()
        for requirement in bundle_run.splitlines()
        if requirement
    ]

    # parse command line arguments
    if '--add' in sys.argv:
        for item in sys.argv[sys.argv.index('--add') + 1 :]:
            if item.startswith('-'):
                break
            EXTRA_REQS.append(item)

    for item in EXTRA_REQS:
        _base = re.split('<|>|=', item, maxsplit=1)[0]
        for r in requirements:
            if r.startswith(_base):
                requirements.remove(r)
                break
        if _base.lower().startswith('pyqt5'):
            try:
                i = next(x for x in requirements if x.startswith('PySide'))
                requirements.remove(i)
            except StopIteration:
                pass

    requirements += EXTRA_REQS

    toml['tool']['briefcase']['app'][APP]['requires'] = requirements
    toml['tool']['briefcase']['version'] = VERSION

    print("patching pyproject.toml to version: ", VERSION)
    print(
        "patching pyproject.toml requirements to:",
        *toml['tool']['briefcase']['app'][APP]['requires'],
        sep="\n ",
    )

    if MACOS:
        # Workaround https://github.com/napari/napari/issues/2965
        # Pin revisions to releases _before_ they switched to static libs
        revision = {
            (3, 6): '11',
            (3, 7): '5',
            (3, 8): '4',
            (3, 9): '1',
        }[sys.version_info[:2]]
        app_table = toml['tool']['briefcase']['app'][APP]
        app_table.add('macOS', tomlkit.table())
        app_table['macOS']['support_revision'] = revision
        print(
            "patching pyproject.toml to pin support package revision:",
            revision,
        )
        # See https://github.com/napari/napari/pull/5152/#issuecomment-1263385953
        # and following comments - we can't use `template_branch` because that
        # is only available in a later briefcase version; we pin to 0.3.1 due to
        # https://github.com/napari/napari/pull/2980
        # so we can only use 'template', which supports git urls, archive urls and
        # paths; we can't use direct URLs because the launcher loses the
        # executable bits (chmod+x)... so we need to clone and checkout
        # ourselves, and then provide a path :/
        # normally we would use the pyver and revision from the dict above, but 3.9b1
        # has a bug and... whatever comes before b9 works with the pydantic bug :shrug:
        # b1 to b4: crash with no error
        # b5 to b8: "Distribution {name!r} exists but does not provide a napari manifest"
        #   note it does work if launched with the bundled python3 -m napari approach
        #   (PYTHONPATH needs to be patched though)
        # b8 and up: pydantic error :/
        template_path = os.path.join(HERE, 'macOS', 'template')
        os.makedirs(template_path, exist_ok=True)
        template_tag = f"3.9-b5"
        print(f"fetchin template {template_tag}...")
        subprocess.check_output(
            [
                "git",
                "clone",
                "--branch",
                template_tag,
                "https://github.com/beeware/briefcase-macOS-app-template",
                template_path,
            ]
        )
        app_table['macOS']['template'] = template_path
        print(
            "patching pyproject.toml to pin template to:",
            template_path,
        )

    with open(PYPROJECT_TOML, 'w') as f:
        f.write(tomlkit.dumps(toml))

    try:
        yield
    finally:
        with open(PYPROJECT_TOML, 'w') as f:
            f.write(original_toml)


@contextmanager
def patched_dmgbuild():
    if not MACOS:
        yield
    else:
        from dmgbuild import core

        with open(core.__file__) as f:
            src = f.read()
        with open(core.__file__, 'w') as f:
            f.write(
                src.replace(
                    "shutil.rmtree(os.path.join(mount_point, '.Trashes'), True)",
                    "shutil.rmtree(os.path.join(mount_point, '.Trashes'), True);time.sleep(30)",
                )
            )
        print("patched dmgbuild.core")
        try:
            yield
        finally:
            # undo
            with open(core.__file__, 'w') as f:
                f.write(src)


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
        with open(fname) as f:
            source = f.read()
        with open(fname, 'w') as f:
            f.write(source.replace('pythonw.exe', 'python.exe'))
            print("patched pythonw.exe -> python.exe")


def patch_python_lib_location():
    # must run after briefcase create
    support = os.path.join(
        BUILD_DIR, APP, APP + ".app", "Contents", "Resources", "Support"
    )
    python_resources = os.path.join(support, "Python", "Resources")
    if os.path.exists(python_resources):
        return
    os.makedirs(python_resources, exist_ok=True)
    for subdir in ("bin", "lib"):
        orig = os.path.join(support, subdir)
        dest = os.path.join(python_resources, subdir)
        os.symlink("../../" + subdir, dest)
        print("symlinking", orig, "to", dest)


def add_sentinel_file():
    if MACOS:
        (Path(APP_DIR) / "Contents" / "MacOS" / ".napari_is_bundled").touch()
    elif LINUX:
        (Path(APP_DIR) / "usr" / "bin" / ".napari_is_bundled").touch()
    elif WINDOWS:
        (Path(APP_DIR) / "python" / ".napari_is_bundled").touch()
    else:
        print("!!! Sentinel files not yet implemented in", sys.platform)


def patch_environment_variables():
    os.environ["ARCH"] = ARCH


def make_zip():
    import glob
    import zipfile

    artifact = glob.glob(os.path.join(BUILD_DIR, f"*.{EXT}"))[0]
    dest = f'napari-{VERSION}-{OS}-{ARCH}.zip'

    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact, arcname=os.path.basename(artifact))
    print("created zipfile: ", dest)
    return dest


def clean():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def bundle():
    clean()

    if LINUX:
        patch_environment_variables()

    # smoke test, and build resources
    subprocess.check_call([sys.executable, '-m', APP, '--info'])

    # the briefcase calls need to happen while the pyproject toml is patched
    with patched_toml(), patched_dmgbuild():
        # create
        cmd = ['briefcase', 'create', '-v'] + (
            ['--no-docker'] if LINUX else []
        )
        subprocess.check_call(cmd)

        time.sleep(0.5)

        add_site_packages_to_path()
        add_sentinel_file()

        if WINDOWS:
            patch_wxs()
        elif MACOS:
            patch_python_lib_location()

        # build
        cmd = ['briefcase', 'build', '-v'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # package
        cmd = ['briefcase', 'package', '-v']
        cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # compress
        dest = make_zip()
        clean()

        return dest


if __name__ == "__main__":
    if '--clean' in sys.argv:
        clean()
        sys.exit()
    if '--version' in sys.argv:
        print(VERSION)
        sys.exit()
    if '--arch' in sys.argv:
        print(ARCH)
        sys.exit()
    print('created', bundle())
