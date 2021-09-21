"""
Create a Napari bundle using `conda` packages and self-installing
mechanisms to userspace.

Steps:

1. Parse setup.cfg and obtain a conda-compatible environment.yml.
2. Use a dry-run in micromamba to obtain the JSON output of the solved environment.
3. Download the packages and check the MD5 hashes.
4. Package everything in Briefcase, along with the launcher logic.
"""

import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from distutils.spawn import find_executable
from pathlib import Path

import tomlkit

from bundle import (
    APP,
    APP_DIR,
    BUILD_DIR,
    LINUX,
    MACOS,
    PYPROJECT_TOML,
    VERSION,
    WINDOWS,
    add_sentinel_file,
    architecture,
    clean,
    make_zip,
    patch_environment_variables,
    patch_python_lib_location,
    patch_wxs,
    patched_dmgbuild,
)


def _generate_conda_build_recipe():
    pass


def _conda_build():
    pass


def _micromamba(root=None, with_local=False, version=VERSION):
    micromamba = find_executable("micromamba")
    if not micromamba:
        raise RuntimeError("Micromamba must be installed and in PATH.")

    if root is None:
        root = tempfile.mkdtemp()

    output = subprocess.check_output(
        [micromamba, "create", "--always-copy", "-y", "-r", root, "-n", "napari"]
        + (["-c", "local"] if with_local else [])
        + [
            "-c",
            "conda-forge",
            f"python={sys.version_info.major}.{sys.version_info.minor}.*",
            "pip",
            "conda",
            "mamba",
            f"napari={version}.*",
        ],
        universal_newlines=True,
    )
    with open("micromamba.log", "w") as out:
        out.write(output)

    # Remove some stuff we do not need in the conda env:
    # - Package cache (tarballs)
    # - Another napari installation coming from conda-forge
    shutil.rmtree(Path(root) / "pkgs")
    napari_env = Path(root) / "envs" / "napari"
    site_packages = (
        napari_env
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    shutil.rmtree(site_packages / "napari")
    for dist_info in site_packages.glob("napari-*.dist-info"):
        shutil.rmtree(dist_info)

    return str(napari_env)


@contextmanager
def _patched_toml():
    with open(PYPROJECT_TOML) as f:
        original_toml = f.read()

    toml = tomlkit.parse(original_toml)
    toml['tool']['briefcase']['version'] = VERSION
    print("patching pyproject.toml to version: ", VERSION)

    if MACOS:
        # Workaround https://github.com/napari/napari/issues/2965
        # Pin revisions to releases _before_ they switched to static libs
        revision = {
            (3, 6): 'b11',
            (3, 7): 'b5',
            (3, 8): 'b4',
            (3, 9): 'b1',
        }[sys.version_info[:2]]
        app_table = toml['tool']['briefcase']['app'][APP]
        app_table.add('macOS', tomlkit.table())
        app_table['macOS']['support_revision'] = revision
        print(
            "patching pyproject.toml to pin support package to revision:",
            revision,
        )

    with open(PYPROJECT_TOML, 'w') as f:
        f.write(tomlkit.dumps(toml))

    try:
        yield
    finally:
        with open(PYPROJECT_TOML, 'w') as f:
            f.write(original_toml)


def main():
    print("Cleaning...")
    clean()

    with_local = False
    if not "release":  # TODO: implement actual checks for non-final releases
        _generate_conda_build_recipe()
        _conda_build()
        with_local = True
    # else: we just build a bundle of the last release in conda-forge
    version = "0.4.11"  # hardcoded now for testing purposes

    print("Debugging info...")

    # smoke test, and build resources
    subprocess.check_call([sys.executable, '-m', APP, '--info'])

    print("Patching runtime conditions...")

    if LINUX:
        patch_environment_variables()

    with _patched_toml(), patched_dmgbuild():
        # create
        cmd = ['briefcase', 'create'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        if WINDOWS:
            patch_wxs()
        elif MACOS:
            patch_python_lib_location()

        add_sentinel_file()

        print("Installing conda environment...")
        root = (
            Path(APP_DIR) / "Contents" / "Resources" / "conda"
            if MACOS
            else Path(BUILD_DIR) / "conda"
        )
        root.mkdir(parents=True, exist_ok=True)
        _micromamba(root, with_local=with_local, version=version)

        # build
        cmd = ['briefcase', 'build'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # package
        cmd = ['briefcase', 'package']
        cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # compress
        print("Creating zipfile...")
        dest = make_zip()
        # clean()

        return dest


if __name__ == "__main__":
    if '--clean' in sys.argv:
        clean()
        sys.exit()
    if '--version' in sys.argv:
        print(VERSION)
        sys.exit()
    if '--arch' in sys.argv:
        print(architecture())
        sys.exit()
    print('created', main())
