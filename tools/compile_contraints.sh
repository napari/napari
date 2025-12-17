#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}/.."  # Move to repo root

flags=(--quiet --extra pyqt6 --extra pyside6 --extra testing --group testing_extra --extra all_optional)

# Explanation of below commands
# uv pip compile --python-version 3.9 - call uv pip compile but ensure proper interpreter
# --upgrade upgrade to the latest possible version. Without this pip-compile will take a look to output files and reuse versions (so will ad something on when adding dependency.
# -o resources/constraints/constraints_py3.9.txt - output file
# pyproject.toml resources/constraints/version_denylist.txt - source files. the resources/constraints/version_denylist.txt - contains our test specific constraints like pytes-cov`
#
# --extra pyqt6 etc - names of extra sections from pyproject.toml that should be checked for the dependencies list (maybe we could create a super extra section to collect them all in)
pyproject_toml="pyproject.toml"
constraints="resources/constraints"


for pyv in 3.10 3.11 3.12 3.13; do
uv pip compile --python-version ${pyv} --upgrade --output-file ${constraints}/constraints_py${pyv}.txt  ${pyproject_toml} ${constraints}/version_denylist.txt "${flags[@]}"
done


uv pip compile --python-version 3.12 --upgrade --output-file ${constraints}/constraints_py3.12_examples.txt ${pyproject_toml} ${constraints}/version_denylist.txt ${constraints}/version_denylist_examples.txt --group gallery "${flags[@]}"
uv pip compile --python-version 3.12 --upgrade --output-file ${constraints}/constraints_py3.12_docs.txt ${pyproject_toml} ${constraints}/version_denylist.txt ${constraints}/version_denylist_examples.txt --group docs "${flags[@]}"
uv pip compile --python-version 3.12 --upgrade --output-file resources/requirements_mypy.txt resources/requirements_mypy.in
