#!/usr/bin/env python3
"""
This script reads your project's pyproject.toml (PEP 621 format) and iterates over all
dependencies (both main and extras). For each dependency that is a binary dependency
(i.e. one that provides platform-specific wheels), it checks whether the version specified
as the minimal (from a lower-bound like ">=1.3.0") provides a wheel compatible with a given
Python version (e.g. 3.10).

If not, it searches (in ascending order) for the lowest version (above the specified minimal)
that provides a compatible wheel and reports the problematic packages with the recommended version bump.

Requirements:
  - Python 3.11+ uses the built-in tomllib; older versions require the external 'toml' package.
  - Install requests and packaging (and toml if needed).

Usage: Run this script in your project root (where pyproject.toml is located).

It is chatGPT generated code.
"""

import re
import sys

import requests

# Use built-in tomllib if available; else fallback to external toml
try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

# --- Helper Functions ---


def parse_min_python_version(requires_python: str):
    """
    Extracts the minimal Python version from a requires-python string (e.g. ">=3.10").
    Assumes a specifier like ">=X.Y" is present.
    """
    m = re.search(r'>=\s*([\d\.]+)', requires_python)
    if m:
        ver_str = m.group(1)
        try:
            return Version(ver_str)
        except InvalidVersion:
            print(f'Could not parse version from requires-python: {ver_str}')
            return None
    else:
        print("No '>=' specifier found in requires-python.")
        return None


def get_specified_min_version(specifier: SpecifierSet) -> Version:
    """
    Returns the effective minimal version from a dependency's specifier set.
    This is the maximum among all lower-bound constraints (">=" or "==").
    """
    lower_bounds = []
    for spec in specifier:
        if spec.operator in ('>=', '=='):
            try:
                lower_bounds.append(Version(spec.version))
            except InvalidVersion:
                continue
    if not lower_bounds:
        return None
    return max(lower_bounds)


def wheel_supports_python(filename: str, required_py_ver: Version) -> bool:
    """
    Determines if the wheel (given by its filename) supports the required Python version.

    Heuristic:
      - Universal wheels (py3-none-any) are accepted.
      - For CPython wheels:
          • If the ABI tag contains "abi3" (e.g. cp38-abi3), assume it works for any CPython version
            that is equal to or newer than the tagged version.
          • Otherwise, the wheel's tag (e.g. "cp310") must match the required version exactly.
    """
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as e:
        print(f'Error parsing wheel filename {filename}: {e}')
        return False

    for tag in tags:
        interpreter = tag.interpreter.lower()
        abi = tag.abi.lower()
        platform = tag.platform.lower()

        # Accept universal pure-Python wheels.
        if interpreter == 'py3' and abi == 'none' and platform == 'any':
            return True

        if interpreter.startswith('cp'):
            ver_str = interpreter[2:]  # e.g. "310" or "39"
            try:
                if len(ver_str) == 2:
                    wheel_major = int(ver_str[0])
                    wheel_minor = int(ver_str[1])
                elif len(ver_str) >= 3:
                    wheel_major = int(ver_str[0])
                    wheel_minor = int(ver_str[1:])
                else:
                    continue
            except ValueError:
                continue

            req_major = required_py_ver.major
            req_minor = required_py_ver.minor

            if 'abi3' in abi:
                # Assume compatibility if the required version is >= the tagged version.
                if (
                    req_major == wheel_major and req_minor >= wheel_minor
                ) or req_major > wheel_major:
                    return True
            else:
                if req_major == wheel_major and req_minor == wheel_minor:
                    return True
    return False


def has_compatible_wheel(files, required_py_ver: Version) -> bool:
    """
    Returns True if any wheel file in the given list is compatible with the required Python version.
    """
    for file_info in files:
        if file_info.get('packagetype') == 'bdist_wheel':
            fname = file_info.get('filename')
            if fname and wheel_supports_python(fname, required_py_ver):
                return True
    return False


def check_version_for_compatible_wheel(
    pkg_name: str, version: Version, required_py_ver: Version
) -> bool:
    """
    Checks whether the given version of a package provides a wheel compatible with the required Python version.
    """
    url = f'https://pypi.org/pypi/{pkg_name}/{version}/json'
    resp = requests.get(url)
    if resp.status_code != 200:
        print(
            f'  Could not fetch data for {pkg_name} version {version} (status {resp.status_code}).'
        )
        return False
    data = resp.json()
    files = data.get('urls', [])
    return has_compatible_wheel(files, required_py_ver)


def find_lowest_version_with_compatible_wheel(
    pkg_name: str,
    spec: SpecifierSet,
    pypi_data: dict,
    required_py_ver: Version,
    current_min: Version,
):
    """
    Among all candidate versions (satisfying the specifier and greater than current_min),
    returns the lowest version that provides a compatible wheel for the required Python version.
    Returns None if none found.
    """
    candidates = []
    releases = pypi_data.get('releases', {})
    for ver_str in releases:
        try:
            ver = Version(ver_str)
        except InvalidVersion:
            continue
        if ver in spec and ver > current_min:
            candidates.append(ver)
    if not candidates:
        return None
    candidates.sort()  # ascending order
    for ver in candidates:
        files = releases.get(str(ver), [])
        if has_compatible_wheel(files, required_py_ver):
            return ver
    return None


def process_dependency(dep_str: str, required_py_ver: Version, origin: str):
    """
    Processes a single dependency string (from main dependencies or extras):
      - Determines the specified minimal version.
      - Checks if that version provides a compatible wheel for the required Python version.
      - If not, searches for the lowest version above the specified minimal version that does.
    Returns a tuple (pkg_name, specified_min, alternative_version) if an issue is found; otherwise None.
    """
    try:
        req_obj = Requirement(dep_str)
    except InvalidRequirement as e:
        print(f"  Failed to parse dependency '{dep_str}': {e}")
        return None

    pkg_name = req_obj.name
    spec = req_obj.specifier or SpecifierSet()
    specified_min = get_specified_min_version(spec)
    if specified_min is None:
        print(
            f'  {pkg_name} ({origin}) has no explicit lower-bound; skipping check.'
        )
        return None

    print(
        f'  {pkg_name} ({origin}): specified minimal version = {specified_min}'
    )

    if check_version_for_compatible_wheel(
        pkg_name, specified_min, required_py_ver
    ):
        print(
            f'    OK: Version {specified_min} provides a compatible wheel for Python {required_py_ver}'
        )
        return None
    # Query PyPI to look for the lowest version above the specified minimal that provides a compatible wheel.
    url = f'https://pypi.org/pypi/{pkg_name}/json'
    resp = requests.get(url)
    if resp.status_code != 200:
        print(
            f'    Could not fetch data for {pkg_name} from PyPI (status {resp.status_code}).'
        )
        return (pkg_name, specified_min, None)
    pypi_data = resp.json()
    alternative = find_lowest_version_with_compatible_wheel(
        pkg_name, spec, pypi_data, required_py_ver, specified_min
    )
    if alternative:
        print(
            f'    ISSUE: {pkg_name}=={specified_min} does NOT provide a compatible wheel; consider bumping to at least version {alternative}'
        )
    else:
        print(
            f'    ISSUE: No version of {pkg_name} satisfying {spec} provides a compatible wheel for Python {required_py_ver}'
        )
    return (pkg_name, specified_min, alternative)


def process_all_dependencies(
    dependencies, required_py_ver: Version, origin: str
):
    """
    Processes a list of dependency strings (from either main dependencies or extras)
    and returns a list of issue reports.
    """
    issues = []
    for dep in dependencies:
        result = process_dependency(dep, required_py_ver, origin)
        if result:
            issues.append(result)
    return issues


def process_extras(extras: dict, required_py_ver: Version):
    """
    Processes extra dependencies (grouped by extra name) and returns a list of issues.
    """
    issues = []
    for extra_name, dep_list in extras.items():
        print(f'\n--- Processing extra: {extra_name} ---')
        issues.extend(
            process_all_dependencies(
                dep_list, required_py_ver, f"extra '{extra_name}'"
            )
        )
    return issues


# --- Main Script ---


def main(pyproject_path='pyproject.toml'):
    try:
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:  # noqa: BLE001
        print(f'Error reading {pyproject_path}: {e}')
        sys.exit(1)

    project = config.get('project')
    if not project:
        print('No [project] section found in pyproject.toml.')
        sys.exit(1)

    requires_python = project.get('requires-python')
    if not requires_python:
        print('No requires-python field found in [project].')
        sys.exit(1)

    req_py_version = parse_min_python_version(requires_python)
    if req_py_version is None:
        print('Could not determine required Python version.')
        sys.exit(1)

    print(f'Required Python minimal version: {req_py_version}')

    overall_issues = []

    # Process main dependencies (list of strings)
    main_deps = project.get('dependencies', [])
    if main_deps:
        print('\n--- Checking main dependencies ---')
        overall_issues.extend(
            process_all_dependencies(main_deps, req_py_version, 'main')
        )
    else:
        print('No main dependencies found.')

    # Process extras (dictionary: extra name -> list of dependencies)
    extras = project.get('optional-dependencies', {})
    if extras:
        print('\n--- Checking extra dependencies ---')
        overall_issues.extend(process_extras(extras, req_py_version))
    else:
        print('No extra dependencies found.')

    if overall_issues:
        print(
            f'\nSummary of problematic binary dependencies (for required Python version {req_py_version}):'
        )
        for pkg_name, spec_min, alternative in overall_issues:
            if alternative:
                print(
                    f'  - {pkg_name}: {pkg_name}=={spec_min} does NOT provide a compatible wheel; consider bumping to at least {alternative}'
                )
            else:
                print(
                    f'  - {pkg_name}: {pkg_name}=={spec_min} does NOT provide a compatible wheel and no alternative was found'
                )
    else:
        print(
            '\nAll binary dependencies provide a compatible wheel for the required Python version based on their specified minimal version.'
        )


if __name__ == '__main__':
    main()
