"""
Create a Napari bundle using `conda` packages and self-installing
mechanisms to userspace.

Steps:

1. Parse setup.cfg and obtain a conda-compatible environment.yml.
2. Use a dry-run in micromamba to obtain the JSON output of the solved environment.
3. Download the packages and check the MD5 hashes.
4. Package everything in Briefcase, along with the launcher logic.
"""

import sys
import re
from distutils.spawn import find_executable
import subprocess
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path

import ruamel_yaml as yaml
import tomlkit

from bundle import (
    APP,
    WINDOWS,
    MACOS,
    LINUX,
    HERE,
    PYPROJECT_TOML,
    SETUP_CFG,
    ARCH,
    BUILD_DIR,
    APP_DIR,
    VERSION,
    patched_dmgbuild,
    patch_environment_variables,
    patch_python_lib_location,
    patch_wxs,
    clean,
    add_site_packages_to_path,
    make_zip,
    add_sentinel_file,
)


def _clean_pypi_spec(spec):
    # remove comments and python selectors
    spec = spec.split("#")[0].split(";")[0].strip()
    # remove [extras] syntax
    match = re.search(r"\S+(\[\S+\]).*", spec)
    if match:
        return spec.replace(match.groups()[0], "")
    # no spaces between constrains
    spec = spec.replace(", ", ",")
    # add space before version requirements
    return spec


def _generate_conda_build_recipe():
    pass


def _conda_build():
    pass


def _constructor(with_local=False, version=VERSION):
    constructor = find_executable("constructor")
    if not constructor:
        raise RuntimeError("Constructor must be installed.")
    micromamba = find_executable("micromamba")

    output_filename = f"bundle.{'exe' if WINDOWS else 'sh'}"
    definitions = {
        "name": APP,
        "company": "Napari",
        "version": f"{version}",
        "channels": (["local"] if with_local else []) + ["conda-forge"],
        "batch_mode": True,
        "installer_filename": output_filename,
        "specs": [
            f"napari={version}",
            f"python={sys.version_info.major}.{sys.version_info.minor}.*",
            "conda",
            "mamba",
            "pip",
        ],
        "exclude": [
            "napari",
        ],
    }

    with open("construct.yaml", "w") as fin:
        yaml.dump(definitions, fin, default_flow_style=False)
        print("-----")
        subprocess.check_call(
            [constructor] + (["--conda-exe", micromamba] if micromamba else []) + ["."],
        )
        print("-----")

    return output_filename


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
    clean()
    print("Generating constructor installer...")
    with_local = False
    if not "release":  # TODO: implement actual checks for non-final releases
        _generate_conda_build_recipe()
        _conda_build()
        with_local = True
    # else: we just build a bundle of the last release in conda-forge
    version = "0.4.11"  # hardcoded now for testing purposes
    constructor_bundle = _constructor(with_local=with_local, version=version)

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
        shutil.move(
            constructor_bundle, Path(APP_DIR) / "Contents" / "Resources" if MACOS else BUILD_DIR
        )

        # build
        cmd = ['briefcase', 'build'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # package
        cmd = ['briefcase', 'package']
        cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
