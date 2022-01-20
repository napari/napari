"""
Check state of vendored modules.
"""

import shutil
import sys
from pathlib import Path
from subprocess import check_output


TOOLS_PATH = Path(__file__).parent
REPO_ROOT_PATH = TOOLS_PATH.parent
VENDOR_FOLDER = "_vendor"
NAPARI_FOLDER = "napari"


def check_vendored_module(org : str, reponame : str, tag : str) -> str:
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
    repo_path = TOOLS_PATH / reponame
    if repo_path.is_dir():
        shutil.rmtree(repo_path)

    check_output(["git", "clone", f"https://github.com/{org}/{reponame}"], cwd=TOOLS_PATH)
    check_output(["git", "checkout", tag], cwd=repo_path)

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
    for org, reponame, tag in [("albertosottile", "darkdetect", "master")]:
        print(f"\n * Checking '{org}/{reponame}'\n")
        diff = check_vendored_module(org, reponame, tag)
        if CI:
            print(f"::set-output name=vendored::{org}/{reponame}")
            sys.exit(0)
        if diff:
            print(diff)
            print(f"\n * '{org}/{reponame}' vendor code seems to not be up to date!!!\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
