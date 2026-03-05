#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}/.."  # Move to repo root

# Decide what to pass to uv pip compile: either a global --upgrade (no args)
# or one or more --upgrade-package <pkg> for packages that are present in
# our constraint files under resources/constraints/*.txt
if [ "$#" -eq 0 ]; then
  upgrade_flag=(--upgrade)
else
  # collect constraint files (use nullglob-like handling)
  constraint_files=( "resources/constraints"/*.txt )

  # start with empty list of upgrade-package pairs
  upgrade_flag=()

  # If there are no constraint files treat it as an error (fail fast)
  if [ "${#constraint_files[@]}" -eq 0 ]; then
    echo "Error: no constraint files found in resources/constraints" >&2
    exit 1
  fi

  # create a temporary concatenated file of all constraints to simplify searching
  tmp_constraints=$(mktemp)
  trap 'rm -f "$tmp_constraints"' EXIT
  cat "${constraint_files[@]}" > "$tmp_constraints"

  # collect missing packages so we can report them all at once
  missing_pkgs=()

  for pkg in "$@"; do
    # Escape regex-special characters in package name for the grep pattern
    safe_pkg=$(printf '%s' "$pkg" | sed 's/[]\\[\.^$*+?(){}|\/]/\\&/g')
    alt_pkg=$(printf '%s' "$safe_pkg" | sed 's/_/-/g')

    # Pattern: start of line (optional whitespace), package name or its alt, then
    # either end-of-token, extras [ , or separators like = < > ~ or whitespace or @ (vcs).
    pattern="^[[:space:]]*(${safe_pkg}|${alt_pkg})(\[|[[:space:]]|=|<|>|~|@)"

    if grep -E -i -q -- "${pattern}" "$tmp_constraints"; then
      upgrade_flag+=(--upgrade-package "$pkg")
    else
      echo "Notice: package '$pkg' not found in resources/constraints/*.txt" >&2
      missing_pkgs+=("$pkg")
    fi
  done

  if [ "${#missing_pkgs[@]}" -ne 0 ]; then
    echo "Error: the following packages were not found in resources/constraints/*.txt: ${missing_pkgs[*]}" >&2
    exit 1
  fi
fi

set -x

# Explanation of below commands
# uv pip compile --python-version 3.9 - call uv pip compile but ensure proper interpreter
# --upgrade upgrade to the latest possible version. Without this pip-compile will take a look to output files and reuse versions (so will ad something on when adding dependency.
# -o resources/constraints/constraints_py3.9.txt - output file
# pyproject.toml resources/constraints/version_denylist.txt - source files. the resources/constraints/version_denylist.txt - contains our test specific constraints like pytes-cov`
#
# --extra pyqt6 etc - names of extra sections from pyproject.toml that should be checked for the dependencies list (maybe we could create a super extra section to collect them all in)
pyproject_toml="pyproject.toml"
constraints="resources/constraints"
flags=(--quiet --extra pyqt6 --extra pyside6 --extra testing --group testing_extra --extra all_optional --exclude ${constraints}/napari_exclude.txt)

for pyv in 3.10 3.11 3.12 3.13 3.14; do
uv pip compile --python-version ${pyv} --output-file ${constraints}/constraints_py${pyv}.txt "${upgrade_flag[@]}" ${pyproject_toml} ${constraints}/version_denylist.txt "${flags[@]}"
done


uv pip compile --python-version 3.12 --output-file ${constraints}/constraints_py3.12_examples.txt "${upgrade_flag[@]}" ${pyproject_toml} ${constraints}/version_denylist.txt ${constraints}/version_denylist_examples.txt --group gallery "${flags[@]}"
uv pip compile --python-version 3.12 --output-file ${constraints}/constraints_py3.12_docs.txt "${upgrade_flag[@]}" ${pyproject_toml} ${constraints}/version_denylist.txt ${constraints}/version_denylist_examples.txt --group docs "${flags[@]}"
uv pip compile --python-version 3.12 --output-file resources/requirements_mypy.txt "${upgrade_flag[@]}" resources/requirements_mypy.in
