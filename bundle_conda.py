"""
Create napari installers using `constructor`.

It creates a `construct.yaml` file with the needed settings
and then runs `constructor`.

For more information, see Documentation> Developers> Packaging.

Some environment variables we use:

CONSTRUCTOR_APP_NAME:
    in case you want to build a non-default distribution that is not
    named `napari`
CONSTRUCTOR_TARGET_PLATFORM:
    conda-style platform (as in `platform` in `conda info -a` output)
CONSTRUCTOR_USE_LOCAL:
    whether to use the local channel (populated by `conda-build` actions)
CONSTRUCTOR_CONDA_EXE:
    when the target platform is not the same as the host, constructor
    needs a path to a conda-standalone (or micromamba) executable for
    that platform. needs to be provided in this env var in that case!
CONSTRUCTOR_SIGNING_IDENTITY:
    Apple ID Installer Certificate identity (common name) that should
    be use to productsign the resulting PKG (macOS only)
CONSTRUCTOR_NOTARIZATION_IDENTITY:
    Apple ID Developer Certificate identity (common name) that should
    be use to codesign some binaries bundled in the pkg (macOS only)
CONSTRUCTOR_SIGNING_CERTIFICATE:
    Path to PFX certificate to sign the EXE installer on Windows
CONSTRUCTOR_PFX_CERTIFICATE_PASSWORD:
    Password to unlock the PFX certificate. This is not used here but
    it might be needed by constructor.
"""

import json
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser
from distutils.spawn import find_executable
from pathlib import Path
from textwrap import indent

from ruamel import yaml

APP = os.environ.get("CONSTRUCTOR_APP_NAME", "napari")
HERE = os.path.abspath(os.path.dirname(__file__))
WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")
if os.environ.get("CONSTRUCTOR_TARGET_PLATFORM") == "osx-arm64":
    ARCH = "arm64"
else:
    ARCH = (platform.machine() or "generic").lower().replace("amd64", "x86_64")
if WINDOWS:
    EXT, OS = 'exe', 'Windows'
elif LINUX:
    EXT, OS = 'sh', 'Linux'
elif MACOS:
    EXT, OS = 'pkg', 'macOS'
else:
    raise RuntimeError(f"Unrecognized OS: {sys.platform}")


def _version():
    with open(os.path.join(HERE, "napari", "_version.py")) as f:
        match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
        if match:
            return match.groups()[0].split('+')[0]


OUTPUT_FILENAME = f"{APP}-{_version()}-{OS}-{ARCH}.{EXT}"
clean_these_files = []


def _use_local():
    """
    Detect whether we need to build Napari locally
    (dev snapshots). This env var is set in the GHA workflow.
    """
    return os.environ.get("CONSTRUCTOR_USE_LOCAL")


def _constructor(version=_version(), extra_specs=None):
    """
    Create a temporary `construct.yaml` input file and
    run `constructor`.

    Parameters
    ----------
    version: str
        Version of `napari` to be built. Defaults to the
        one detected by `setuptools-scm` and written to
        `napari/_version.py`. Run `pip install -e .` to
        generate that file if it can't be found.
    extra_specs: list of str
        Additional packages to be included in the installer.
        A list of conda spec strings (`python`, `python=3`, etc)
        is expected.
    """
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")

    if extra_specs is None:
        extra_specs = []

    # TODO: Temporary while pyside2 is not yet published for arm64
    target_platform = os.environ.get("CONSTRUCTOR_TARGET_PLATFORM")
    ARM64 = target_platform == "osx-arm64"
    if ARM64:
        napari = f"napari={version}=*pyqt*"
    else:
        napari = f"napari={version}=*pyside*"
    specs = [
        napari,
        f"napari-menu={version}",
        f"python={sys.version_info.major}.{sys.version_info.minor}.*",
        "conda",
        "mamba",
        "pip",
    ] + extra_specs

    channels = (
        ["napari/label/nightly"]
        + (["andfoy"] if ARM64 else [])  # TODO: temporary
        + ["napari/label/bundle_tools", "conda-forge"]
    )
    definitions = {
        "name": APP,
        "company": "Napari",
        "reverse_domain_identifier": "org.napari",
        "version": version,
        "channels": channels,
        "conda_default_channels": ["conda-forge"],
        "installer_filename": OUTPUT_FILENAME,
        "initialize_by_default": False,
        "license_file": os.path.join(HERE, "resources", "bundle_license.rtf"),
        "specs": specs,
        "menu_packages": [
            "napari-menu",
        ],
        "extra_files": {"resources/bundle_readme.md": "README.txt"},
    }
    if _use_local():
        definitions["channels"].insert(0, "local")
    if LINUX:
        definitions["default_prefix"] = os.path.join(
            "$HOME", ".local", f"{APP}-{version}"
        )
        definitions["license_file"] = os.path.join(
            HERE, "resources", "bundle_license.txt"
        )

    if MACOS:
        # we change this bc the installer takes the name
        # as the default install location basename
        definitions["name"] = f"{APP}-{version}"
        definitions["default_location_pkg"] = "Library"
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
        notarization_identity = os.environ.get(
            "CONSTRUCTOR_NOTARIZATION_IDENTITY"
        )
        if notarization_identity:
            definitions["notarization_identity_name"] = notarization_identity

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
                "default_prefix": os.path.join(
                    '%LOCALAPPDATA%', f"{APP}-{version}"
                ),
                "default_prefix_domain_user": os.path.join(
                    '%LOCALAPPDATA%', f"{APP}-{version}"
                ),
                "default_prefix_all_users": os.path.join(
                    '%ALLUSERSPROFILE%', f"{APP}-{version}"
                ),
                "check_path_length": False,
            }
        )
        signing_certificate = os.environ.get("CONSTRUCTOR_SIGNING_CERTIFICATE")
        if signing_certificate:
            definitions["signing_certificate"] = signing_certificate

    clean_these_files.append("construct.yaml")

    # TODO: temporarily patching password - remove block when the secret has been fixed
    # (I think it contains an ending newline or something like that, copypaste artifact?)
    pfx_password = os.environ.get("CONSTRUCTOR_PFX_CERTIFICATE_PASSWORD")
    if pfx_password:
        os.environ[
            "CONSTRUCTOR_PFX_CERTIFICATE_PASSWORD"
        ] = pfx_password.strip()

    with open("construct.yaml", "w") as fin:
        yaml.dump(definitions, fin, default_flow_style=False)

    args = [constructor, "-v", "--debug", "."]
    conda_exe = os.environ.get("CONSTRUCTOR_CONDA_EXE")
    if target_platform and conda_exe:
        args += ["--platform", target_platform, "--conda-exe", conda_exe]
    env = os.environ.copy()
    env["CONDA_CHANNEL_PRIORITY"] = "strict"

    print(f"Calling {args} with these definitions:")
    print(yaml.dump(definitions, default_flow_style=False))
    subprocess.check_call(args, env=env)

    return OUTPUT_FILENAME


def licenses():
    try:
        with open("info.json") as f:
            info = json.load(f)
    except FileNotFoundError:
        print(
            "!! Use `constructor --debug` to write info.json and get licenses"
        )
        return

    for package_id, license_info in info["_licenses"].items():
        print("\n+++++++++++++++++++++\n")
        for license_type, license_files in license_info.items():
            print(package_id, "=", license_type, "\n")
            for license_file in license_files:
                with open(license_file, "rb") as f:
                    print(indent(f.read().decode(errors="ignore"), "    "))


def main(extra_specs=None):
    try:
        _constructor(extra_specs=extra_specs)
    finally:
        for path in clean_these_files:
            os.unlink(path)
    assert Path(OUTPUT_FILENAME).exists()
    return OUTPUT_FILENAME


def cli(argv=None):
    p = ArgumentParser(argv)
    p.add_argument(
        "--version",
        action="store_true",
        help="Print local napari version and exit.",
    )
    p.add_argument(
        "--arch",
        action="store_true",
        help="Print machine architecture tag and exit.",
    )
    p.add_argument(
        "--ext",
        action="store_true",
        help="Print installer extension for this platform and exit.",
    )
    p.add_argument(
        "--artifact-name",
        action="store_true",
        help="Print computed artifact name and exit.",
    )
    p.add_argument(
        "--extra-specs",
        nargs="+",
        help="One or more extra conda specs to add to the installer",
    )
    p.add_argument(
        "--licenses",
        action="store_true",
        help="Post-process licenses AFTER having built the installer. "
        "This must be run as a separate step.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    if args.version:
        print(_version())
        sys.exit()
    if args.arch:
        print(ARCH)
        sys.exit()
    if args.ext:
        print(EXT)
        sys.exit()
    if args.artifact_name:
        print(OUTPUT_FILENAME)
        sys.exit()
    if args.licenses:
        licenses()
        sys.exit()

    print('created', main(extra_specs=args.extra_specs))
