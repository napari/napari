"""
Create a Napari bundle using `conda` packages and self-installing
mechanisms to userspace.

Steps:

1. Parse setup.cfg and obtain a conda-compatible environment.yml.
2. Use a dry-run in micromamba to obtain the JSON output of the solved environment.
3. Download the packages and check the MD5 hashes.
4. Package everything in Briefcase, along with the launcher logic.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from distutils.spawn import find_executable
from pathlib import Path

import tomlkit
from ruamel import yaml

from bundle import (
    APP,
    APP_DIR,
    ARCH,
    BUILD_DIR,
    EXT,
    HERE,
    LINUX,
    MACOS,
    OS,
    PYPROJECT_TOML,
    VERSION,
    WINDOWS,
    add_sentinel_file,
    clean,
    make_zip,
    patch_environment_variables,
    patch_python_lib_location,
    patch_wxs,
    patched_dmgbuild,
)

if LINUX:
    CONDA_ROOT = Path(APP_DIR) / "usr" / "conda"
    EXT = "AppImage"  # using briefcase
elif MACOS:
    CONDA_ROOT = Path(APP_DIR) / "Contents" / "Resources" / "conda"
    EXT = "dmg"  # using briefcase
elif WINDOWS:
    CONDA_ROOT = Path(APP_DIR) / "conda"
    EXT = "exe"  # using constructor
else:
    CONDA_ROOT = Path(BUILD_DIR) / "conda"


VERSION = "0.4.11"  # overwriting for testing purposes
OUTPUT_FILENAME = f"{APP}-{VERSION}-{OS}-{ARCH}.{EXT}"


def _use_local():
    """
    Detect whether we need to build Napari locally
    (dev snapshots).
    """


def _generate_conda_build_recipe():
    pass


def _conda_build():
    pass


def _constructor(version=VERSION):
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")
    micromamba = os.environ.get("MAMBA_EXE", find_executable("micromamba"))

    definitions = {
        "name": APP,
        "company": "Napari",
        "version": version,
        "channels": ["conda-forge"],
        "conda_default_channels": ["conda-forge"],
        "installer_filename": OUTPUT_FILENAME,
        "specs": [
            f"napari={version}",
            f"python={sys.version_info.major}.{sys.version_info.minor}.*",
            "conda",
            "mamba",
            "pip",
        ],
        "menu_packages": [
            "napari",
        ],
    }
    if _use_local():
        definitions["channels"].insert(0, "local")
    if MACOS:
        definitions["installer_type"] = "pkg"
    if WINDOWS:
        definitions["conda_default_channels"].append("defaults")
        definitions.update(
            {
                # TODO: create banner images for installer
                # "welcome_image":,
                # "header_image":,
                "icon_image": os.path.join(HERE, "resources", "icon.ico"),
                "default_image_color": "blue",
                "welcome_image_text": f"{APP} v{version}",
                "header_image_text": f"{APP} v{version}",
            }
        )

    with open("construct.yaml", "w") as fin:
        yaml.dump(definitions, fin, default_flow_style=False)
        print("-----")
        subprocess.check_call(
            [constructor] + (["--conda-exe", micromamba] if micromamba else []) + ["."],
        )
        print("-----")

    return OUTPUT_FILENAME


def _micromamba(root=None, version=VERSION):
    micromamba = os.environ.get("MAMBA_EXE", find_executable("micromamba"))
    if not micromamba:
        raise RuntimeError("Micromamba must be installed and in PATH.")

    if root is None:
        root = tempfile.mkdtemp()

    output = subprocess.check_output(
        [
            micromamba,
            "create",
            "--always-copy",
            "-y",
            "-r",
            root,
            "-n",
            "napari",
        ]
        + (["-c", "local"] if _use_local() else [])
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
    shutil.rmtree(Path(root) / "pkgs", ignore_errors=True)
    napari_env = Path(root) / "envs" / "napari"
    if WINDOWS:
        napari_env = napari_env / "Library"

    site_packages = (
        napari_env
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    shutil.rmtree(site_packages / "napari", ignore_errors=True)
    for dist_info in site_packages.glob("napari-*.dist-info"):
        shutil.rmtree(dist_info, ignore_errors=True)

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

    if not "release":  # TODO: implement actual checks for non-final releases
        _generate_conda_build_recipe()
        _conda_build()

    print("Debugging info...")

    # smoke test, and build resources
    subprocess.check_call([sys.executable, '-m', APP, '--info'])

    if WINDOWS:
        _constructor()
    else:
        _briefcase()

    assert Path(OUTPUT_FILENAME).exists()
    return OUTPUT_FILENAME


def _briefcase(version=VERSION):
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
        CONDA_ROOT.mkdir(parents=True, exist_ok=True)
        _micromamba(CONDA_ROOT, version=version)

        # build
        cmd = ['briefcase', 'build'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # package
        cmd = ['briefcase', 'package']
        cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # Rename to desired artifact name
        artifact = next(Path(BUILD_DIR).glob(f"*.{EXT}"))
        print(f"Renaming {artifact.name} to {OUTPUT_FILENAME}")
        artifact.rename(OUTPUT_FILENAME)


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
    if '--ext' in sys.argv:
        print(EXT)
        sys.exit()
    if '--artifact-name' in sys.argv:
        print(OUTPUT_FILENAME)
        sys.exit()
    print('created', main())
