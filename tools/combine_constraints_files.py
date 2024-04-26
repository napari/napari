import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from packaging.version import Version, parse as parse_version


class PackageInfo(NamedTuple):
    package_name: str
    version: Version
    comments: list[str]
    python_version: Version
    os_str: str


NAPARI_PATH = Path(__file__).absolute().parent.parent

constraint_pydantic_1 = re.compile(
    r'constraints_py(?P<python>3.\d{1,2})_(?P<os_str>.+)_pydantic_1.txt'
)
constraint_pydantic_2 = re.compile(
    r'constraints_py(?P<python>3.\d{1,2})_(?P<os_str>.+).txt'
)


def load_constraints(file_path: Path, python: str, os_str: str):
    python_version = parse_version(python)
    result_constraints = {}
    current_package = None
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.strip().startswith('#'):
                continue
            current_package, version = line.strip().split('==')
            result_constraints[current_package] = PackageInfo(
                current_package,
                parse_version(version),
                [],
                python_version,
                os_str,
            )
            break

        for line in f:
            if not line.strip():
                continue
            if line.strip().startswith('#'):
                result_constraints[current_package].comments.append(
                    line.strip()
                )
                continue
            current_package, version = line.strip().split('==')
            result_constraints[current_package] = PackageInfo(
                current_package,
                parse_version(version),
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
            result[package][info.version].append(info)
    return result


def combine_constraint_entry(
    name: str, versions: dict[Version, list[PackageInfo]]
):
    if len(versions) == 1:
        constraints_str = f'{name}=={next(iter(versions.keys()))}'
        comments = set()
        for package_info in next(iter(versions.values())):
            comments.update(package_info.comments)
        if comments:
            constraints_str += '\n   ' + '\n   '.join(comments)
        return constraints_str
    return f'# {name}=={next(iter(versions.keys()))}'


def generate_constraints_file(
    constraints: dict[str, dict[Version, list[PackageInfo]]],
):
    reuslt = []
    for package, versions in constraints.items():
        reuslt.append(combine_constraint_entry(package, versions))

    return '\n'.join(reuslt)


OS_MAPPING = {
    'x86_64-apple-darwin': 1,
    'aarch64-apple-darwin': 1,
    'windows': 1,
    'x86_64-manylinux_2_28': 1,
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
