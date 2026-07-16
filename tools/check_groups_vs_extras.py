"""
This script is called in pre-commit to ensure that certain extras in
pyproject.toml are kept in sync with dependency-groups.

It is to provide a transition period while we migrate from extras to
dependency-groups, and can be removed once the migration is complete.

It is planned to remote these extras and this script in the 0.7.0 release.
"""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    sys.exit('This script requires Python 3.11 or newer.')


SYNC_GROUPS = (
    'testing',
    'testing_extra',
    'gallery',
    'docs',
    'release',
    'dev',
    'build',
)


def include_group_to_extras(dependency: str | dict) -> str:
    if isinstance(dependency, str):
        return dependency
    return f'napari[{dependency["include-group"]}]'


def main():
    pyproject_toml_path = Path(__file__).parent.parent / 'pyproject.toml'
    data = tomllib.loads(pyproject_toml_path.read_text())
    extras = data['project']['optional-dependencies']
    groups = data['dependency-groups']

    for group_name in SYNC_GROUPS:
        group = {include_group_to_extras(x) for x in groups[group_name]}
        extra = set(extras[group_name])
        if group != extra:
            print(f"Mismatch in '{group_name}':")
            only_in_group = group - extra
            only_in_extra = extra - group
            if only_in_group:
                print(f'  In group but not in extras: {only_in_group}')
            if only_in_extra:
                print(f'  In extras but not in group: {only_in_extra}')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
