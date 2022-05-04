"""
Create napari installers using `constructor`.

It creates a `construct.yaml` file with the needed settings
and then runs `constructor`.

For more information, see Documentation> Developers> Packaging.

Some environment variables we use:

CONSTRUCTOR_APP_NAME:
    in case you want to build a non-default distribution that is not
    named `napari`
CONSTRUCTOR_NAPARI_VERSION:
    version of napari you want to build the installer for. If not provided,
    it expects the repository to be installed in development mode so
    `$REPO_ROOT/napari/_version.py` is populated.
CONSTRUCTOR_INSTALLER_VERSION:
    Version for the installer, separate from the app being installed.
    This has an effect on the default install locations!
CONSTRUCTOR_INSTALLER_DEFAULT_PATH_STEM:
    The last component of the default installation path. Defaults to
    {CONSTRUCTOR_APP_NAME}-app-{CONSTRUCTOR_INSTALLER_VERSION}
CONSTRUCTOR_TARGET_PLATFORM:
    conda-style platform (as in `platform` in `conda info -a` output)
CONSTRUCTOR_PYTHON_VERSION:
    Version of Python to ship in the installer, with `major.minor` syntax
    (e.g. 3.8). If not provided, will default to the version of the
    interpreter running this script.
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

import configparser
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
from textwrap import dedent

from ruamel import yaml

APP = os.environ.get("CONSTRUCTOR_APP_NAME", "napari")
# bump this when something in the installer infrastructure changes
# note that this will affect the default installation path across platforms!
INSTALLER_VERSION = os.environ.get("CONSTRUCTOR_INSTALLER_VERSION", "0.1")
INSTALLER_DEFAULT_PATH_STEM = os.environ.get(
    "CONSTRUCTOR_INSTALLER_DEFAULT_PATH_STEM", f"{APP}-app-{INSTALLER_VERSION}"
)
HERE = os.path.abspath(os.path.dirname(__file__))
WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")
PYTHON_VERSION = os.environ.get(
    "CONSTRUCTOR_PYTHON_VERSION",
    f"{sys.version_info.major}.{sys.version_info.minor}",
)
TARGET_PLATFORM = os.environ.get("CONSTRUCTOR_TARGET_PLATFORM")
if TARGET_PLATFORM == "osx-arm64":
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


def _get_version():
    if "CONSTRUCTOR_NAPARI_VERSION" in os.environ:
        return os.environ["CONSTRUCTOR_NAPARI_VERSION"]

    with open(os.path.join(HERE, "napari", "_version.py")) as f:
        match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
        if match:
            return match.groups()[0].split('+')[0]


OUTPUT_FILENAME = f"{APP}-{_get_version()}-{OS}-{ARCH}.{EXT}"
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


def _patch_napari_recipe(recipe_path: str):
    from souschef.jinja_expression import set_global_jinja_var
    from souschef.recipe import Recipe

    recipe = Recipe(load_file=recipe_path)

    # 1. patch version
    set_global_jinja_var(recipe, "version", _get_version())

    # 2. patch sources
    del recipe["source"][-1]
    del recipe["source"][0]["url"]
    del recipe["source"][0]["sha256"]
    recipe["source"][0].update(
        # this path is crafted to match the source path inside the docker
        # image that builds the conda package
        {"path": "/home/conda/feedstock_root/napari-source"}
    )

    # 3. patch run requirements for napari pkg
    napari_host_reqs = recipe["outputs"][0]["requirements"]["host"]
    napari_run_reqs = recipe["outputs"][0]["requirements"]["run"]
    napari_run_reqs.clear()
    for spec, comment in _get_dependencies()["napari_recipe"]:
        napari_run_reqs.append(spec)
        if "python" in spec.split():
            for idx, req in enumerate(napari_host_reqs):
                if "python" in req.value.split():
                    break
            else:
                raise ValueError("python not found in host?")
            napari_host_reqs[idx].value = spec
        if comment:
            napari_run_reqs[-1].inline_comment = comment

    # 4. add new napari-pinnings output
    new_output = {
        "name": "napari-pinnings",
        "version": _get_version(),
        "build": {
            "noarch": "generic",
            "skip": True,
        },
        "number": "<{ build }}",
        "requirements": {"run_constrained": []},
        "about": {
            "home": "http://napari.org",
            "license": "BSD-3-Clause",
            "license_family": "BSD",
            "license_file": "LICENSE",
            "summary": "provides pinnings used by the bundle installer",
            "doc_url": "http://napari.org",
            "dev_url": "https://github.com/napari/napari",
        },
    }

    recipe["outputs"].append("")  # work around .append() limitation
    recipe["outputs"][2] = new_output

    for spec, comment in _get_dependencies()["run_constrained"]:
        constrains = recipe["outputs"][2]["requirements"]["run_constrained"]
        constrains.append(spec)
        if comment:
            constrains.inline_comment = comment

    skip = recipe["outputs"][2]["build"]["skip"]
    skip.inline_comment = "[qt_bindings == 'pyside2']"

    recipe.save(recipe_path)

    return recipe


def _lines_from_cfg_block(block: str, comments=False):
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split("#", 1)]

        # conda specs can have a "selector" comment:  # [linux]
        # we need to keep that around for the yaml syntax!
        # no comment? add a blank field so every line has two fields
        if comments:
            if len(fields) == 1:
                fields.append("")
            lines.append(fields)
        else:
            lines.append(fields[0])
    return lines


def _get_dependencies():
    # TODO: Temporary while pyside2 is not yet published for arm64
    napari_build_str = "*pyqt*" if ARCH == "arm64" else "*pyside*"
    napari_version_str = _get_version()
    python_version_str = f"={PYTHON_VERSION}.*"
    cfg = configparser.ConfigParser()
    cfg.read("setup.cfg")

    base_channels = _lines_from_cfg_block(
        cfg["conda_installer"]["base_run_channels"]
    )
    base_specs = _lines_from_cfg_block(cfg["conda_installer"]["base_run"])
    base_specs[base_specs.index("python")] += python_version_str

    napari_specs = _lines_from_cfg_block(cfg["conda_installer"]["napari_run"])
    napari_idx = napari_specs.index("napari")
    napari_specs[napari_idx] += f"={napari_version_str}={napari_build_str}"
    napari_menu_idx = napari_specs.index("napari-menu")
    napari_specs[napari_menu_idx] += f"={napari_version_str}"

    napari_channels = []
    if _use_local():
        napari_channels.append("local")
    if ARCH == "arm64":
        # temporary workaround for missing packages
        napari_channels.append("andfoy")
    napari_channels += _lines_from_cfg_block(
        cfg["conda_installer"]["napari_run_channels"]
    )

    menu_specs = _lines_from_cfg_block(
        cfg["conda_installer"]["napari_run_shortcuts"]
    )

    napari_pinnings = _lines_from_cfg_block(
        cfg["conda_installer"]["napari_run_constrained"],
        comments=True,
    )
    napari_recipe_run = _lines_from_cfg_block(
        cfg["conda_installer"]["napari_recipe_run"],
        comments=True,
    )

    return {
        "base": base_specs,
        "base_channels": base_channels,
        "napari": napari_specs,
        "napari_channels": napari_channels,
        "menu_packages": menu_specs,
        "run_constrained": napari_pinnings,
        "napari_recipe": napari_recipe_run,
    }


def _get_condarc():
    # we need defaults for tensorflow and others on windows only
    defaults = "- defaults" if WINDOWS else ""
    prompt = "[napari]({default_env}) "
    contents = dedent(
        f"""
        channels:  #!final
          - napari
          - conda-forge
          {defaults}
        repodata_fns:  #!final
          - repodata.json
        auto_update_conda: false  #!final
        channel_priority: strict  #!final
        env_prompt: '{prompt}'  #! final
        """
    )
    # the undocumented #!final comment is explained here
    # https://www.anaconda.com/blog/conda-configuration-engine-power-users
    with NamedTemporaryFile(delete=False, mode="w+") as f:
        f.write(contents)
    return f.name


def _constructor():
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
    """
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")

    version = _get_version()
    dependencies = _get_dependencies()

    empty_file = NamedTemporaryFile(delete=False)
    condarc = _get_condarc()
    definitions = {
        "name": APP,
        "company": "Napari",
        "reverse_domain_identifier": "org.napari",
        "version": version,
        "channels": dependencies["base_channels"],
        "conda_default_channels": ["conda-forge"],
        "installer_filename": OUTPUT_FILENAME,
        "initialize_by_default": False,
        "license_file": os.path.join(HERE, "resources", "bundle_license.rtf"),
        "specs": dependencies["base"],
        "extra_envs": {
            f"napari-{version}": {
                "specs": dependencies["napari"],
                "channels": dependencies["napari_channels"],
            },
        },
        "menu_packages": dependencies["menu_packages"],
        "extra_files": {
            "resources/bundle_readme.md": "README.txt",
            empty_file.name: ".napari_is_bundled_constructor",
            condarc: ".condarc",
        },
    }
    if LINUX:
        definitions["default_prefix"] = os.path.join(
            "$HOME", ".local", INSTALLER_DEFAULT_PATH_STEM
        )
        definitions["license_file"] = os.path.join(
            HERE, "resources", "bundle_license.txt"
        )
        definitions["installer_type"] = "sh"

    if MACOS:
        # These two options control the default install location:
        # ~/<default_location_pkg>/<pkg_name>
        definitions["pkg_name"] = INSTALLER_DEFAULT_PATH_STEM
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
                    '%LOCALAPPDATA%', INSTALLER_DEFAULT_PATH_STEM
                ),
                "default_prefix_domain_user": os.path.join(
                    '%LOCALAPPDATA%', INSTALLER_DEFAULT_PATH_STEM
                ),
                "default_prefix_all_users": os.path.join(
                    '%ALLUSERSPROFILE%', INSTALLER_DEFAULT_PATH_STEM
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
    clean_these_files.append(condarc)

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
    if TARGET_PLATFORM and conda_exe:
        args += ["--platform", TARGET_PLATFORM, "--conda-exe", conda_exe]
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


def main():
    try:
        _constructor()
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
        "--installer-version",
        action="store_true",
        help="Print installer version and exit.",
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
    p.add_argument(
        "--patch-recipe",
        type=str,
        help="Patch conda-forge recipe for our internal CI usage",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    if args.version:
        print(_get_version())
        sys.exit()
    if args.installer_version:
        print(INSTALLER_VERSION)
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
    if args.patch_recipe:
        _patch_napari_recipe(args.patch_recipe)
        sys.exit()

    print('created', main())
