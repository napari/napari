import argparse
import re
import subprocess  # nosec
import sys
from configparser import ConfigParser
from pathlib import Path

name_re = re.compile(r"[\w-]+")
changed_name_re = re.compile(r"\+([\w-]+)")


src_dir = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--main-packages", action="store_true")
args = parser.parse_args()

out = subprocess.run(  # nosec
    [
        "git",
        "diff",
        str(src_dir / "resources" / "constraints" / "constraints_py3.9.txt"),
    ],
    capture_output=True,
    check=True,
)

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
    print("only non direct updates")
