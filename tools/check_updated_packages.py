import argparse
import os
import re
import subprocess  # nosec
import sys
from configparser import ConfigParser
from pathlib import Path

from get_auto_upgrade_branch_name import get_base_branch_name

name_re = re.compile(r"[\w-]+")
changed_name_re = re.compile(r"\+([\w-]+)")


src_dir = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--main-packages", action="store_true")
args = parser.parse_args()

ref_name = os.environ.get("GITHUB_REF_NAME")
event = os.environ.get("GITHUB_EVENT_NAME", "")
if ref_name is None:
    out = subprocess.run(  # nosec
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        check=True,
    )
    ref_name = out.stdout.decode().strip()

base_branch = get_base_branch_name(ref_name, event)

out = subprocess.run(  # nosec
    ["git", "branch", "--list", "--format", "%(refname:short)", "-a"],
    capture_output=True,
    check=True,
)

branches = out.stdout.decode().split("\n")

if base_branch not in branches:
    if f"origin/{base_branch}" not in branches:
        print(f"base branch {base_branch} not found in {branches!r}")
        sys.exit(1)

    base_branch = f"origin/{base_branch}"


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
    print("stderr", e.stderr.decode())
    print("stdout", e.stdout.decode())
    sys.exit(1)

changed_packages = [
    changed_name_re.match(x)[1].lower()
    for x in out.stdout.decode().split("\n")
    if changed_name_re.match(x)
]

if not args.main_packages:
    print("\n".join(f" * {x}" for x in sorted(changed_packages)))
    sys.exit(0)


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

if updated_core_packages := sorted(set(packages) & set(changed_packages)):
    print(", ".join(f"`{x}`" for x in updated_core_packages))
else:
    print("only indirect updates")
