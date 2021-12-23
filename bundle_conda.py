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
clean_these_files = []


def _use_local():
    """
    Detect whether we need to build Napari locally
    (dev snapshots). This env var is set in the GHA workflow.
    """
    return os.environ.get("CONSTRUCTOR_USE_LOCAL")


def _soft_wrap_text(text: str) -> str:
    """
    Join contiguous non-empty lines into a long line.
    This replaces hard-wrapped text with its soft-wrap
    equivalent.
    """
    lines = []
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line.strip())
        else:
            paragraphs.append(" ".join(lines))
            lines = []

    return "\n\n".join(paragraphs)


def _license_file():
    # collect license(s)
    license_file = Path(HERE) / "LICENSE"
    text = license_file.read_text()
    if MACOS:
        # PKG installer looks weird if linebreaks are kept
        text = _soft_wrap_text(text)

    # write to file
    license_out = Path(HERE) / "processed_license.txt"
    license_out.write_text(text)
    clean_these_files.append(license_out)
    return str(license_out)


def _constructor(version=VERSION):
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")

    definitions = {
        "name": APP,
        "company": "Napari",
        "reverse_domain_identifier": "org.napari",
        "version": version,
        "channels": [
            "napari/label/nightly",
            "napari/label/bundle_tools",
            "conda-forge",
        ],
        "conda_default_channels": ["conda-forge"],
        "installer_filename": OUTPUT_FILENAME,
        "initialize_by_default": False,
        "license_file": _license_file(),
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
    if LINUX:
        definitions["default_prefix"] = os.path.join('%USERPROFILE%', f"{APP}-{version}")
    if MACOS:
        # we change this bc the installer takes the name as the default install location basename
        definitions["name"] = f"{APP}-{version}"
        definitions["installer_type"] = "pkg"
        definitions["welcome_image"] = os.path.join(
            HERE, "resources", "napari_1227x600.png"
        )
        welcome_text_tmpl = (
            Path(HERE) / "resources" / "osx_pkg_welcome.rtf.tmpl"
        ).read_text()
        welcome_file = Path(HERE) / "resources" / "osx_pkg_welcome.rtf"
        clean_these_files.append(welcome_file)
        welcome_file.write_text(
            welcome_text_tmpl.replace("__VERSION__", version)
        )
        definitions["welcome_file"] = str(welcome_file)
        definitions["conclusion_text"] = ""
        definitions["readme_text"] = ""
        signing_identity = os.environ.get("CONSTRUCTOR_SIGNING_IDENTITY")
        if signing_identity:
            definitions["signing_identity_name"] = signing_identity

    if WINDOWS:
        definitions["conda_default_channels"].append("defaults")
        definitions.update(
            {
                "welcome_image": os.path.join(
                    HERE, "resources", "napari_164x314.png"
                ),
                "header_image": os.path.join(
                    HERE, "resources", "napari_150x57.png"
                ),
                "icon_image": os.path.join(
                    HERE, "napari", "resources", "icon.ico"
                ),
                "register_python_default": False,
                "default_prefix": os.path.join('%USERPROFILE%', f"{APP}-{version}"),
                "default_prefix_domain_user": os.path.join('%LOCALAPPDATA%', f"{APP}-{version}"),
                "default_prefix_all_users": os.path.join('%ALLUSERSPROFILE%', f"{APP}-{version}"),
            }
        )
        signing_certificate = os.environ.get("CONSTRUCTOR_SIGNING_CERTIFICATE")
        if signing_certificate:
            definitions["signing_certificate"] = signing_certificate

    print("Calling `constructor` with these definitions:")
    print(yaml.dump(definitions, default_flow_style=False))
    clean_these_files.append("construct.yaml")

    with open("construct.yaml", "w") as fin:
        yaml.dump(definitions, fin, default_flow_style=False)

    subprocess.check_call([constructor, "-v", "."])

    return OUTPUT_FILENAME


def main():
    print("Cleaning...")
    clean()
    try:
        _constructor()
    finally:
        for path in clean_these_files:
            os.unlink(path)

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
