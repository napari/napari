"""
Check state of vendored modules.
"""

import shutil
import sys
from pathlib import Path
from subprocess import check_output
from typing import List


TOOLS_PATH = Path(__file__).parent
REPO_ROOT_PATH = TOOLS_PATH.parent
VENDOR_FOLDER = "_vendor"
NAPARI_FOLDER = "napari"


def _clone(org, reponame, tag):
    repo_path = TOOLS_PATH / reponame
    if repo_path.is_dir():
        shutil.rmtree(repo_path)

    check_output(
        [
            "git",
            "clone",
            '--depth',
            '1',
            '--branch',
            tag,
            f"https://github.com/{org}/{reponame}",
        ],
        cwd=TOOLS_PATH,
    )

    return repo_path


def check_vendored_files(
    org: str, reponame: str, tag: str, source_paths: List[Path], target_path: Path
) -> str:
    repo_path = _clone(org, reponame, tag)
    vendor_path = REPO_ROOT_PATH / NAPARI_FOLDER / target_path
    for s in source_paths:
        shutil.copy(repo_path / s, vendor_path)
    return check_output(["git", "diff"], cwd=vendor_path).decode("utf-8")


def check_vendored_module(org: str, reponame: str, tag: str) -> str:
    """
    Check if the vendored module is up to date.

    Parameters
    ----------
    org : str
        The github organization name.
    reponame : str
        The github repository name.
    tag : str
        The github tag.

    Returns
    -------
    str
        Returns the diff if the module is not up to date or an empty string
        if it is.
    """
    repo_path = _clone(org, reponame, tag)

    vendor_path = REPO_ROOT_PATH / NAPARI_FOLDER / VENDOR_FOLDER / reponame
    if vendor_path.is_dir():
        shutil.rmtree(vendor_path)

    shutil.copytree(repo_path / reponame, vendor_path)
    shutil.copy(repo_path / "LICENSE", vendor_path)
    shutil.rmtree(repo_path, ignore_errors=True)

    return check_output(["git", "diff"], cwd=vendor_path).decode("utf-8")


def main():
    CI = '--ci' in sys.argv
    print("\n\nChecking vendored modules\n")
    vendored_modules = []
    for org, reponame, tag, source, target in [
        ("albertosottile", "darkdetect", "master", None, None),
        (
            "matplotlib",
            "matplotlib",
            "v3.7.1",
            [
                # this file seem to be post 3.0.3 but pre 3.1
                # plus there may have been custom changes.
                # 'lib/matplotlib/colors.py',
                #
                # this file seem much more recent, but is touched much more rarely.
                # it is at least from 3.2.1 as the turbo colormap is present and
                # was added in matplotlib in 3.2.1
                'lib/matplotlib/_cm_listed.py'
            ],
            'utils/colormaps/vendored/',
        ),
    ]:
        print(f"\n * Checking '{org}/{reponame}'\n")
        if source is None:
            diff = check_vendored_module(org, reponame, tag)
        else:
            diff = check_vendored_files(
                org, reponame, tag, [Path(s) for s in source], Path(target)
            )

        if diff:
            vendored_modules.append((org, reponame, diff))

    if CI:
        with open(TOOLS_PATH / "vendored_modules.txt", "w") as f:
            f.write("\n".join(f"{org}/{reponame}" for org, reponame, _ in vendored_modules))
        sys.exit(0)
    if vendored_modules:
        print("\n\nThe following vendored modules are not up to date:\n")
        for org, reponame, _diff in vendored_modules:
            print(f"\n * {org}/{reponame}\n")
        sys,exit(1)


if __name__ == "__main__":
    main()
