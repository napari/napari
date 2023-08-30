from __future__ import annotations

import argparse
import os
import re
import subprocess  # nosec
import sys
from configparser import ConfigParser
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
DEFAULT_NAME = "auto-dependency-upgrades"


def get_base_branch_name(ref_name, event):
    if ref_name == DEFAULT_NAME:
        return "main"
    if ref_name.startswith(DEFAULT_NAME):
        if event in {"pull_request", "pull_request_target"}:
            return os.environ.get("GITHUB_BASE_REF")
        return ref_name[len(DEFAULT_NAME) + 1 :]
    return ref_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-packages", action="store_true")
    args = parser.parse_args()

    ref_name = get_ref_name()
    event = os.environ.get("GITHUB_EVENT_NAME", "")

    base_branch = get_base_branch_name(ref_name, event)

    try:
        res = get_changed_dependencies(base_branch, not args.main_packages)
    except ValueError as e:
        print(e)
        sys.exit(1)

    if args.main_packages:
        print("\n".join(f" * {x}" for x in sorted(res)))
    elif res:
        print(", ".join(f"`{x}`" for x in res))
    else:
        print("only indirect updates")


def get_branches() -> list[str]:
    """
    Get all branches from the repository.
    """
    out = subprocess.run(  # nosec
        ["git", "branch", "--list", "--format", "%(refname:short)", "-a"],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode().split("\n")


def calc_changed_packages(base_branch: str, src_dir: Path) -> list[str]:
    """
    Calculate a list of changed packages based on python 3.10

    Parameters
    ----------
    base_branch: str
        branch against which to compare
    src_dir: Path
        path to the root of the repository

    Returns
    -------
    list[str]
        list of changed packages
    """
    changed_name_re = re.compile(r"\+([\w-]+)")

    try:
        out = subprocess.run(  # nosec
            [
                "git",
                "diff",
                base_branch,
                str(
                    src_dir
                    / "resources"
                    / "constraints"
                    / "constraints_py3.10.txt"
                ),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"git diff failed with return code {e.returncode}"
            " stderr: {e.stderr.decode()!r}"
            " stdout: {e.stdout.decode()!r}"
        ) from e

    return [
        changed_name_re.match(x)[1].lower()
        for x in out.stdout.decode().split("\n")
        if changed_name_re.match(x)
    ]


def get_ref_name() -> str:
    """
    Get the name of the current branch.
    """
    ref_name = os.environ.get("GITHUB_REF_NAME")
    if ref_name:
        return ref_name
    out = subprocess.run(  # nosec
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode().strip()


def calc_only_direct_updates(
    changed_packages: list[str], src_dir: Path
) -> list[str]:
    name_re = re.compile(r"[\w-]+")

    config = ConfigParser()
    config.read(src_dir / "setup.cfg")
    packages = (
        config["options"]["install_requires"].split("\n")
        + config["options.extras_require"]["pyqt5"].split("\n")
        + config["options.extras_require"]["pyqt6_experimental"].split("\n")
        + config["options.extras_require"]["pyside2"].split("\n")
        + config["options.extras_require"]["pyside6_experimental"].split("\n")
        + config["options.extras_require"]["testing"].split("\n")
        + config["options.extras_require"]["all"].split("\n")
    )
    packages = [
        name_re.match(package).group().lower()
        for package in packages
        if name_re.match(package)
    ]
    return sorted(set(packages) & set(changed_packages))


def get_changed_dependencies(base_branch: str, all_packages=False):
    """
    Get the changed dependencies.

    all_packages: bool
        If True, return all packages, not just the direct dependencies.
    """
    src_dir = Path(__file__).parent.parent

    branches = get_branches()

    if base_branch not in branches:
        if f"origin/{base_branch}" not in branches:
            raise ValueError(
                f"base branch {base_branch} not found in {branches!r}"
            )
        base_branch = f"origin/{base_branch}"

    changed_packages = calc_changed_packages(base_branch, src_dir)

    if all_packages:
        return sorted(set(changed_packages))

    return calc_only_direct_updates(changed_packages, src_dir)


if __name__ == "__main__":
    main()
