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
import zipfile
from argparse import ArgumentParser
from distutils.spawn import find_executable
from pathlib import Path
from tempfile import NamedTemporaryFile

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


def _generate_background_images(installer_type, outpath="resources"):
    if installer_type == "sh":
        # shell installers are text-based, no graphics
        return

    from PIL import Image

    import napari

    logo_path = Path(napari.__file__).parent / "resources" / "logo.png"
    logo = Image.open(logo_path, "r")

    global clean_these_files

    if installer_type in ("exe", "all"):
        sidebar = Image.new("RGBA", (164, 314), (0, 0, 0, 0))
        sidebar.paste(logo.resize((101, 101)), (32, 180))
        output = Path(outpath, "napari_164x314.png")
        sidebar.save(output, format="png")
        clean_these_files.append(output)

        banner = Image.new("RGBA", (150, 57), (0, 0, 0, 0))
        banner.paste(logo.resize((44, 44)), (8, 6))
        output = Path(outpath, "napari_150x57.png")
        banner.save(output, format="png")
        clean_these_files.append(output)

    if installer_type in ("pkg", "all"):
        background = Image.new("RGBA", (1227, 600), (0, 0, 0, 0))
        background.paste(logo.resize((148, 148)), (95, 418))
        output = Path(outpath, "napari_1227x600.png")
        background.save(output, format="png")
        clean_these_files.append(output)


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
    base_specs = [
        f"python={sys.version_info.major}.{sys.version_info.minor}.*",
        "conda",
        "mamba",
        "pip",
    ]
    napari_specs = [
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
    empty_file = NamedTemporaryFile(delete=False)
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
        "specs": base_specs,
        "extra_envs": {f"napari-{version}": {"specs": napari_specs}},
        "menu_packages": [
            "napari-menu",
        ],
        "extra_files": {
            "resources/bundle_readme.md": "README.txt",
            empty_file.name: ".napari_is_bundled_constructor",
        },
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
        definitions["installer_type"] = "sh"

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
                "installer_type": "exe",
            }
        )
        signing_certificate = os.environ.get("CONSTRUCTOR_SIGNING_CERTIFICATE")
        if signing_certificate:
            definitions["signing_certificate"] = signing_certificate

    if definitions.get("welcome_image") or definitions.get("header_image"):
        _generate_background_images(
            definitions.get("installer_type", "all"), outpath="resources"
        )

    clean_these_files.append("construct.yaml")
    clean_these_files.append(empty_file.name)

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
            "!! Use `constructor --debug` to write info.json and get licenses",
            file=sys.stderr,
        )
        raise

    zipname = f"licenses.{OS}-{ARCH}.zip"
    output_zip = zipfile.ZipFile(
        zipname, mode="w", compression=zipfile.ZIP_DEFLATED
    )
    output_zip.write("info.json")
    for package_id, license_info in info["_licenses"].items():
        package_name = package_id.split("::", 1)[1]
        for license_type, license_files in license_info.items():
            for i, license_file in enumerate(license_files, 1):
                arcname = (
                    f"{package_name}.{license_type.replace(' ', '_')}.{i}.txt"
                )
                output_zip.write(license_file, arcname=arcname)
    output_zip.close()
    return zipname


def main(extra_specs=None):
    try:
        _constructor(extra_specs=extra_specs)
    finally:
        for path in clean_these_files:
            try:
                os.unlink(path)
            except OSError:
                print("! Could not remove", path)
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
    p.add_argument(
        "--images",
        action="store_true",
        help="Generate background images from the logo (test only)",
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
        print(licenses())
        sys.exit()
    if args.images:
        _generate_background_images()
        sys.exit()

    print('created', main(extra_specs=args.extra_specs))
