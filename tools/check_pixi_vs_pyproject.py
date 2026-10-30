"""
This script is called in pre-commit to ensure that dependencies in pixi.toml
are kept in sync with pyproject.toml.

pixi-build-python translates `project.dependencies` to conda packages
automatically, but some parts need to be written manually in pixi.toml:

* `package.extra-dependencies` - mirror of `project.optional-dependencies`
  for Qt backends,
* `package.run-dependencies` - packages missing in the PyPI to conda-forge mapping,
* `feature.test.dependencies` - mirror of the `testing` dependency group
  and the `optional-numba` extra,
* `feature.test.target.<platform>.dependencies` - platform specific
  `project.dependencies` (skipped because napari is built as a noarch package).

Requirements with environment markers are evaluated for each pixi platform
and Python version. As pixi.toml cannot express platform specific extras,
version specifiers of a package from all matching requirements are merged.
"""

import sys
from itertools import product
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    sys.exit('This script requires Python 3.11 or newer.')

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

PROJECT_NAME = 'napari'

# pixi feature mirroring testing dependencies and its sources in pyproject.toml
PIXI_TEST_FEATURE = 'test'
TEST_GROUPS = ('testing',)
TEST_EXTRAS = ('optional-numba',)
# packages used in tests that are not listed in pyproject.toml (see tox.ini)
PIXI_ONLY_TEST_PACKAGES = {'pytest-json-report', 'objgraph'}

# PyPI name (canonical) -> conda-forge name (canonical)
PYPI_TO_CONDA = {
    'matplotlib': 'matplotlib-base',
    'pyqt6-qt6': 'qt6-main',
    'qtconsole': 'qtconsole-base',
}
# (PyPI name, extra) -> additional conda-forge package
PYPI_EXTRA_TO_CONDA = {
    ('lxml', 'html-clean'): 'lxml-html-clean',
}
# conda-forge packages that share the version with Qt, so their version
# exclusions could be expressed by `qt6-main` in `package.run-constraints`
QT_VERSIONED = {'pyside6', 'qt6-main'}

PIXI_PLATFORM_MARKERS = {
    'linux-64': {
        'sys_platform': 'linux',
        'platform_system': 'Linux',
        'platform_machine': 'x86_64',
        'os_name': 'posix',
    },
    'linux-aarch64': {
        'sys_platform': 'linux',
        'platform_system': 'Linux',
        'platform_machine': 'aarch64',
        'os_name': 'posix',
    },
    'osx-64': {
        'sys_platform': 'darwin',
        'platform_system': 'Darwin',
        'platform_machine': 'x86_64',
        'os_name': 'posix',
    },
    'osx-arm64': {
        'sys_platform': 'darwin',
        'platform_system': 'Darwin',
        'platform_machine': 'arm64',
        'os_name': 'posix',
    },
    'win-64': {
        'sys_platform': 'win32',
        'platform_system': 'Windows',
        'platform_machine': 'AMD64',
        'os_name': 'nt',
    },
}

Specs = dict[str, set[str]]


def conda_name(name: str) -> str:
    name = canonicalize_name(name)
    return PYPI_TO_CONDA.get(name, name)


def spec_clauses(spec: str | dict) -> set[str]:
    """Convert a pixi/PEP 440 version specifier to a set of normalized clauses."""
    if isinstance(spec, dict):
        spec = spec.get('version', '*')
    spec = spec.strip()
    if spec in ('', '*'):
        return set()
    return {str(x) for x in SpecifierSet(spec)}


def pixi_specs(dependencies: dict) -> Specs:
    return {
        canonicalize_name(k): spec_clauses(v) for k, v in dependencies.items()
    }


def marker_environments(pixi: dict) -> list[dict[str, str]]:
    platforms = pixi['workspace']['platforms']
    if unknown := set(platforms) - PIXI_PLATFORM_MARKERS.keys():
        sys.exit(f'Unknown pixi platforms: {sorted(unknown)}')
    python_versions = sorted(
        {
            feature['dependencies']['python'].rstrip('.*')
            for feature in pixi.get('feature', {}).values()
            if 'python' in feature.get('dependencies', {})
        }
    )
    return [
        {
            **PIXI_PLATFORM_MARKERS[platform],
            'python_version': python_version,
            'python_full_version': f'{python_version}.0',
            'pixi_platform': platform,
        }
        for platform, python_version in product(platforms, python_versions)
    ]


def is_applicable(
    req: Requirement, environments: list[dict[str, str]]
) -> bool:
    return req.marker is None or any(
        req.marker.evaluate(env) for env in environments
    )


def expand(
    dependencies: list,
    pyproject: dict,
) -> list[Requirement]:
    """Expand `napari[...]` and `include-group` references to requirements."""
    extras = pyproject['project']['optional-dependencies']
    groups = pyproject.get('dependency-groups', {})
    result = []
    for dependency in dependencies:
        if isinstance(dependency, dict):
            result.extend(
                expand(groups[dependency['include-group']], pyproject)
            )
            continue
        req = Requirement(dependency)
        if canonicalize_name(req.name) == PROJECT_NAME:
            for extra in sorted(req.extras):
                result.extend(expand(extras[extra], pyproject))
            continue
        result.append(req)
    return result


def expected_specs(
    requirements: list[Requirement], environments: list[dict[str, str]]
) -> Specs:
    specs: Specs = {}
    for req in requirements:
        if not is_applicable(req, environments):
            continue
        name = conda_name(req.name)
        specs.setdefault(name, set()).update(str(x) for x in req.specifier)
        for extra in req.extras:
            key = (canonicalize_name(req.name), canonicalize_name(extra))
            if key in PYPI_EXTRA_TO_CONDA:
                specs.setdefault(PYPI_EXTRA_TO_CONDA[key], set())
    return specs


def compare(
    section: str,
    expected: Specs,
    actual: Specs,
    *,
    allowed_extra: set[str] = frozenset(),
) -> list[str]:
    errors = []
    for name in sorted(expected.keys() - actual.keys()):
        errors.append(f'{section}: missing `{name}` ({_fmt(expected[name])})')
    for name in sorted(actual.keys() - expected.keys() - allowed_extra):
        errors.append(f'{section}: `{name}` is not present in pyproject.toml')
    for name in sorted(expected.keys() & actual.keys()):
        if expected[name] != actual[name]:
            errors.append(
                f'{section}: version mismatch for `{name}`: '
                f'pyproject.toml "{_fmt(expected[name])}", '
                f'pixi.toml "{_fmt(actual[name])}"'
            )
    return errors


def _fmt(clauses: set[str]) -> str:
    return ','.join(sorted(clauses)) or '*'


def check_extras(pyproject: dict, pixi: dict, environments) -> list[str]:
    errors = []
    package = pixi['package']
    extras = pyproject['project']['optional-dependencies']
    run_constraints = pixi_specs(package.get('run-constraints', {}))
    qt_constraints = run_constraints.get('qt6-main', set())

    for extra_name, dependencies in package.get(
        'extra-dependencies', {}
    ).items():
        section = f'[package.extra-dependencies.{extra_name}]'
        if extra_name not in extras:
            errors.append(f'{section}: extra not present in pyproject.toml')
            continue
        expected = expected_specs(
            expand(extras[extra_name], pyproject), environments
        )
        actual = pixi_specs(dependencies)
        for name in expected.keys() & QT_VERSIONED:
            actual[name] = actual.get(name, set()) | qt_constraints
        errors.extend(compare(section, expected, actual))

    for name, feature in pixi.get('feature', {}).items():
        project_dep = feature.get('dependencies', {}).get(PROJECT_NAME)
        if not isinstance(project_dep, dict):
            continue
        for extra in project_dep.get('extras', []):
            if extra not in package.get('extra-dependencies', {}):
                errors.append(
                    f'[feature.{name}.dependencies]: extra `{extra}` is not '
                    'defined in [package.extra-dependencies]'
                )
    return errors


def check_run_dependencies(
    pyproject: dict, pixi: dict, environments
) -> list[str]:
    expected = expected_specs(
        expand(pyproject['project']['dependencies'], pyproject), environments
    )
    actual = pixi_specs(pixi['package'].get('run-dependencies', {}))
    expected = {k: v for k, v in expected.items() if k in actual}
    return compare('[package.run-dependencies]', expected, actual)


def check_platform_dependencies(
    pyproject: dict, pixi: dict, environments
) -> list[str]:
    """Check that platform specific `project.dependencies` are in pixi targets."""
    errors = []
    targets = pixi['feature'][PIXI_TEST_FEATURE].get('target', {})
    requirements = expand(pyproject['project']['dependencies'], pyproject)
    for platform in pixi['workspace']['platforms']:
        platform_envs = [
            e for e in environments if e['pixi_platform'] == platform
        ]
        platform_specific = [
            req
            for req in requirements
            if req.marker is not None
            and is_applicable(req, platform_envs)
            and not all(req.marker.evaluate(env) for env in environments)
        ]
        expected = expected_specs(platform_specific, platform_envs)
        actual = pixi_specs(targets.get(platform, {}).get('dependencies', {}))
        errors.extend(
            compare(
                f'[feature.{PIXI_TEST_FEATURE}.target.{platform}.dependencies]',
                expected,
                actual,
            )
        )
    return errors


def check_test_feature(pyproject: dict, pixi: dict, environments) -> list[str]:
    groups = pyproject['dependency-groups']
    extras = pyproject['project']['optional-dependencies']
    dependencies = [dep for group in TEST_GROUPS for dep in groups[group]]
    dependencies += [dep for extra in TEST_EXTRAS for dep in extras[extra]]
    expected = expected_specs(expand(dependencies, pyproject), environments)
    actual = pixi_specs(
        pixi['feature'][PIXI_TEST_FEATURE].get('dependencies', {})
    )
    return compare(
        f'[feature.{PIXI_TEST_FEATURE}.dependencies]',
        expected,
        actual,
        allowed_extra=PIXI_ONLY_TEST_PACKAGES,
    )


def main():
    base_dir = Path(__file__).parent.parent
    pyproject = tomllib.loads((base_dir / 'pyproject.toml').read_text())
    pixi = tomllib.loads((base_dir / 'pixi.toml').read_text())
    environments = marker_environments(pixi)

    errors = [
        *check_extras(pyproject, pixi, environments),
        *check_run_dependencies(pyproject, pixi, environments),
        *check_platform_dependencies(pyproject, pixi, environments),
        *check_test_feature(pyproject, pixi, environments),
    ]
    if errors:
        print('pixi.toml is not in sync with pyproject.toml:')
        for error in errors:
            print(f'  {error}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
