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

import ruamel_yaml as yaml


APP = "napari"
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")
HERE = os.path.abspath(os.path.dirname(__file__))
PYPROJECT_TOML = os.path.join(HERE, "pyproject.toml")
SETUP_CFG = os.path.join(HERE, "setup.cfg")
ARCH = platform.machine() or "generic"


if WINDOWS:
    BUILD_DIR = os.path.join(HERE, "windows")
elif LINUX:
    BUILD_DIR = os.path.join(HERE, "linux")
elif MACOS:
    BUILD_DIR = os.path.join(HERE, "macOS")
    APP_DIR = os.path.join(BUILD_DIR, APP, f"{APP}.app")

with open(os.path.join(HERE, "napari", "_version.py")) as f:
    match = re.search(r"version\s?=\s?\'([^\']+)", f.read())
    if match:
        VERSION = match.groups()[0].split("+")[0]


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


def _download_packages(packages):
    output = Path(HERE) / "bundle_workspace" / "pkgs"
    output.mkdir(parents=True, exist_ok=True)
    for package in tqdm(packages):
        _download_and_check(package, output)


def _download_and_check(url_md5, target_directory):
    url, md5 = url_md5.split("#")
    filename = url.split("/")[-1]
    dest_path = Path(target_directory) / filename
    with urlopen(url) as fsrc, open(dest_path, "wb") as fdst:
        copyfileobj(fsrc, fdst)

    with open(dest_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

    obtained_md5 = file_hash.hexdigest()
    assert md5 == obtained_md5, (
        f"Expected MD5 hash for URL {url} does not match obtained:\n"
        f"- Expected: {md5}\n"
        f"- Obtained: {obtained_md5}"
    )


def _setup_briefcase_workspace():
    pass


def _package():
    pass


def main():
    environment_definitions = _setup_cfg_to_environment_yml()
    solved_environment = _solve_environment(environment_definitions)
    urls = [f'{pkg["url"]}#{pkg["md5"]}' for pkg in solved_environment["actions"]["FETCH"]]
    _download_packages(urls)
    _setup_briefcase_workspace()
    _package()


if __name__ == "__main__":
    main()
