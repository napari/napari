from __future__ import annotations

import re
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from packaging.requirements import Requirement
from packaging.version import Version, parse as parse_version


class PackageInfo(NamedTuple):
    version: Requirement
    comments: list[str]
    python_version: Version
    os_str: str


class PlatformInfo(NamedTuple):
    machine: None | str
    system: None | str


NAPARI_PATH = Path(__file__).absolute().parent.parent
python_versions_set = set()

constraint_pydantic_1 = re.compile(
    r'constraints_py(?P<python>3.\d{1,2})_(?P<os_str>.+)_pydantic_1.txt'
)
constraint_pydantic_2 = re.compile(
    r'constraints_py(?P<python>3.\d{1,2})_(?P<os_str>.+).txt'
)


def load_constraints(file_path: Path, python: str, os_str: str):
    python_version = parse_version(python)
    python_versions_set.add(python_version)
    result_constraints = {}
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.strip().startswith('#'):
                continue
            package_info = Requirement(line.strip())
            result_constraints[package_info.name] = PackageInfo(
                package_info,
                [],
                python_version,
                os_str,
            )
            break

        for line in f:
            if not line.strip():
                continue
            if line.strip().startswith('#'):
                if (
                    str_info := line.replace('#', '')
                    .replace('via', '')
                    .strip()
                ):
                    result_constraints[package_info.name].comments.append(
                        str_info
                    )
                continue
            package_info = Requirement(line.strip())
            result_constraints[package_info.name] = PackageInfo(
                package_info,
                [],
                python_version,
                os_str,
            )
    return result_constraints


def combine_constraints(
    list_of_constraints: list[dict[str, PackageInfo]],
) -> dict[str, dict[Version, list[PackageInfo]]]:
    result = defaultdict(lambda: defaultdict(list))
    for constraints in list_of_constraints:
        for package, info in constraints.items():
            result[package][str(info.version.specifier)].append(info)
    return result


def comments_to_str(comments: Iterable[str]) -> str:
    comments_ = sorted(comments)
    if len(comments_) == 1:
        return f'\n    # via {next(iter(comments_))}'
    if comments_:
        return ''.join(f'\n    # {x}' for x in comments_)
    return ''


def combine_constraint_entry(
    name: str, versions: dict[Version, list[PackageInfo]]
):
    if len(versions) == 1:
        constraints_str = f'{name}=={next(iter(versions.keys()))}'
        comments = set()
        for package_info in next(iter(versions.values())):
            comments.update(package_info.comments)
        constraints_str += comments_to_str(comments)
        return constraints_str

    # result_str = ''
    #
    # for version, package_infos in versions.items():
    #     if len(package_infos) == 1:
    #         package_info = package_infos[0]
    #         os_info = OS_MAPPING.get(package_info.os_str)
    #         marker = f"platform_system == '{os_info.system}'"
    #         if os_info.machine:
    #             marker += f" and platform_machine == '{os_info.machine}'"
    #         package_info.version.marker = Marker(marker)
    #         result_str += str(package_info.version) + comments_to_str(
    #             package_info.comments
    #         )

    return f'# {name}=={next(iter(versions.keys()))}'


def generate_constraints_file(
    constraints: dict[str, dict[Version, list[PackageInfo]]],
):
    result = []
    for package, versions in constraints.items():
        result.append(combine_constraint_entry(package, versions))

    return '\n'.join(result)


OS_MAPPING = {
    'x86_64-apple-darwin': PlatformInfo('x86_64', 'Darwin'),
    'aarch64-apple-darwin': PlatformInfo('arm64', 'Darwin'),
    'windows': PlatformInfo(None, 'Windows'),
    'x86_64-manylinux_2_28': PlatformInfo(None, 'Linux'),
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'napari_path', type=Path, default=NAPARI_PATH, nargs='?'
    )

    args = parser.parse_args()
    constraints_dir = Path(args.napari_path) / 'resources' / 'constraints'

    pydantic_1_list = []
    pydantic_2_list = []

    for file_path in constraints_dir.glob('constraints_*.txt'):
        if match := constraint_pydantic_1.match(file_path.name):
            pydantic_1_list.append(
                load_constraints(file_path, **match.groupdict())
            )
        elif match := constraint_pydantic_2.match(file_path.name):
            if match.group('os_str') not in OS_MAPPING:
                continue
            pydantic_2_list.append(
                load_constraints(file_path, **match.groupdict())
            )

    print(generate_constraints_file(combine_constraints(pydantic_1_list)))
    # print("#" * 20)
    # print(combine_constraints(pydantic_2_list))


if __name__ == '__main__':
    main()
