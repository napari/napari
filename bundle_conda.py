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
import sys
import platform
import re
import configparser
from distutils.spawn import find_executable
import subprocess
import json
import tempfile
from urllib.request import urlopen
from shutil import copyfileobj
from pathlib import Path
from tqdm import tqdm
import hashlib
from contextlib import contextmanager

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
    patch_dmgbuild,
    undo_patch_dmgbuild,
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


def _setup_cfg_to_environment_yml(selector=None):
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG)
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = [_clean_pypi_spec(r) for r in requirements if r]

    return {
        "name": APP,
        "channels": ["conda-forge"],
        "dependencies": sorted(requirements),
    }


def _solve_environment(environment_dict):
    micromamba = find_executable("micromamba")
    with tempfile.NamedTemporaryFile(suffix=".env.yml", mode="w") as tmp:
        yaml.dump(environment_dict, tmp, default_flow_style=False)
        out = subprocess.check_output(
            [
                micromamba,
                "create",
                "--yes",
                "--dry-run",
                "-n",
                f"{APP}-bundle",
                "--json",
                "-f",
                tmp.name,
            ],
            universal_newlines=True,
        )

    solved = json.loads(out)
    return solved


def _urls_from_solved_environment(solved):
    return [f'{pkg["url"]}#{pkg["md5"]}' for pkg in solved["actions"]["FETCH"]]


def _download_packages(packages, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for package in tqdm(packages):
        _download_and_check(package, target_dir)


def _download_and_check(url_md5, target_directory):
    url, md5 = url_md5.split("#")
    filename = url.split("/")[-1]
    dest_path = Path(target_directory) / filename
    download = True
    if dest_path.exists():
        try:
            _check_md5(dest_path, md5)
            download = False
        except AssertionError:
            download = True
    if download:
        with urlopen(url) as fsrc, open(dest_path, "wb") as fdst:
            copyfileobj(fsrc, fdst)
        _check_md5(dest_path, md5)


def _check_md5(path, md5):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

    obtained_md5 = file_hash.hexdigest()
    assert md5 == obtained_md5, (
        f"Expected MD5 hash for package {path} does not match obtained:\n"
        f"- Expected: {md5}\n"
        f"- Obtained: {obtained_md5}"
    )


def _setup_briefcase_workspace():
    pass


def _package():
    pass


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
    print("Checking if environment is solvable...")
    environment_definitions = _setup_cfg_to_environment_yml()
    solved_environment = _solve_environment(environment_definitions)
    urls = _urls_from_solved_environment(solved_environment)

    print("Patching runtime conditions...")
    if MACOS:
        patch_dmgbuild()

    if LINUX:
        patch_environment_variables()

    with _patched_toml():
        # create
        cmd = ['briefcase', 'create'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        add_sentinel_file()

        # download pkgs to workspace
        print("Downloading packages...")
        target = Path(APP_DIR) / "Contents" / "Resources" / "Support" if MACOS else BUILD_DIR
        _download_packages(urls, target_dir=Path(target) / "pkgs")

        if WINDOWS:
            patch_wxs()
        elif MACOS:
            patch_python_lib_location()

        # build
        cmd = ['briefcase', 'build'] + (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

        # package
        cmd = ['briefcase', 'package']
        cmd += ['--no-sign'] if MACOS else (['--no-docker'] if LINUX else [])
        subprocess.check_call(cmd)

    if MACOS:
        undo_patch_dmgbuild()


if __name__ == "__main__":
    main()
