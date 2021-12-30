"""
Create napari installers using `constructor`.

It creates a `construct.yaml` file with the needed settings
and then runs `constructor`.

For more information, see Documentation> Developers> Packaging.
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from distutils.spawn import find_executable
from pathlib import Path

from ruamel import yaml

from bundle import APP, ARCH, HERE, LINUX, MACOS, OS, VERSION, WINDOWS, clean

if LINUX:
    EXT = "sh"
elif MACOS:
    EXT = "pkg"
elif WINDOWS:
    EXT = "exe"
else:
    raise RuntimeError(f"Unrecognized OS: {sys.platform}")

if os.environ.get("CONSTRUCTOR_TARGET_PLATFORM") == "osx-arm64":
    ARCH = "arm64"

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


def _constructor(version=VERSION, extra_specs=None):
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

    # Temporary while pyside2 is not yet published for arm64
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
        + (["andfoy"] if ARM64 else [])  # temporary
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
        "license_file": _license_file(),
        "specs": specs,
        "menu_packages": [
            "napari-menu",
        ],
    }
    if _use_local():
        definitions["channels"].insert(0, "local")
    if LINUX:
        definitions["default_prefix"] = os.path.join(
            "$HOME", f"{APP}-{version}"
        )
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
                "default_prefix": os.path.join(
                    '%USERPROFILE%', f"{APP}-{version}"
                ),
                "default_prefix_domain_user": os.path.join(
                    '%LOCALAPPDATA%', f"{APP}-{version}"
                ),
                "default_prefix_all_users": os.path.join(
                    '%ALLUSERSPROFILE%', f"{APP}-{version}"
                ),
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

    args = [constructor, "-v", "."]
    conda_exe = os.environ.get("CONSTRUCTOR_CONDA_EXE")
    if target_platform and conda_exe:
        args += ["--platform", target_platform, "--conda-exe", conda_exe]
    env = os.environ.copy()
    env["CONDA_CHANNEL_PRIORITY"] = "strict"
    subprocess.check_call(args, env=env)

    return OUTPUT_FILENAME


def main(extra_specs=None):
    print("Cleaning...")
    clean()
    try:
        _constructor(extra_specs=None)
    finally:
        for path in clean_these_files:
            os.unlink(path)

    assert Path(OUTPUT_FILENAME).exists()
    return OUTPUT_FILENAME


def cli(argv=None):
    p = ArgumentParser()
    p.add_argument(
        "--clean", action="store_true", help="Clean files and exit."
    )
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
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    if args.clean:
        clean()
        sys.exit()
    if args.version:
        print(VERSION)
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

    print('created', main(extra_specs=args.extra_specs))
