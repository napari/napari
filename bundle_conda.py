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
import subprocess
import sys
from distutils.spawn import find_executable
from pathlib import Path

from ruamel import yaml

from bundle import (
    APP,
    APP_DIR,
    ARCH,
    HERE,
    LINUX,
    MACOS,
    OS,
    VERSION,
    WINDOWS,
    clean,
)

if LINUX:
    CONDA_ROOT = Path(APP_DIR) / "usr" / "conda"
    EXT = "sh"  # using constructor
elif MACOS:
    CONDA_ROOT = Path(APP_DIR) / "Contents" / "Resources" / "conda"
    EXT = "pkg"  # using constructor
elif WINDOWS:
    CONDA_ROOT = Path(APP_DIR) / "conda"
    EXT = "exe"  # using constructor
else:
    raise RuntimeError(f"Unrecognized OS: {sys.platform}")


OUTPUT_FILENAME = f"{APP}-{VERSION}-{OS}-{ARCH}.{EXT}"


def _use_local():
    """
    Detect whether we need to build Napari locally
    (dev snapshots). This env var is set in the GHA workflow.
    """
    return os.environ.get("CONSTRUCTOR_USE_LOCAL")


def _constructor(version=VERSION):
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")

    definitions = {
        "name": APP,
        "company": "Napari",
        "version": version,
        "channels": [
            "napari/label/nightly",
            "napari/label/bundle_tools",
            "conda-forge",
        ],
        "conda_default_channels": ["conda-forge"],
        "installer_filename": OUTPUT_FILENAME,
        "initialize_by_default": False,
        "specs": [
            f"napari={version}",
            f"napari-menu={version}",
            f"python={sys.version_info.major}.{sys.version_info.minor}.*",
            "conda",
            "mamba",
            "pip",
        ],
        "menu_packages": [
            "napari-menu",
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
                "icon_image": os.path.join(
                    HERE, "napari", "resources", "icon.ico"
                ),
                "default_image_color": "blue",
                "welcome_image_text": f"{APP}",
                "header_image_text": f"{APP}",
                "register_python_default": False,
            }
        )

    print("Calling `constructor` with these definitions:")
    print(yaml.dump(definitions, default_flow_style=False))

    with open("construct.yaml", "w") as fin:
        yaml.dump(definitions, fin, default_flow_style=False)

    subprocess.check_call([constructor, "-v", "."])

    return OUTPUT_FILENAME


def main():
    print("Cleaning...")
    clean()

    _constructor()

    assert Path(OUTPUT_FILENAME).exists()
    return OUTPUT_FILENAME


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
