#!/bin/bash
set -e

usage() {

  cat <<'EOF'

Usage:
  compile_constraints.sh [REPO_ROOT] [PACKAGE ...]

Description:
  Rebuild the constraint files for the napari/napari repository using uv.

Arguments:
  REPO_ROOT     Optional root path to the napari repository.
                If the first argument is not a directory, the repository's path is set to the parent directory of the script.
  PACKAGE       Optional list of package name(s) to upgrade selectively.

Behaviour:
  If no PACKAGE arguments are given, it will perform a global upgrade.
  All PACKAGE arguments are validated against the constraint files in REPO_ROOT/resources/constraints/*.txt
  The main dependency definitions are read from the pyproject.toml file in REPO_ROOT.
  It will apply additional restrictions from REPO_ROOT/resources/constraints/.
  It will write the version-specific constraints under REPO_ROOT/resources/constraints/.
  It will write the docs/examples constraints under REPO_ROOT/resources/constraints/.
  It will write mypy requirements under REPO_ROOT/resources/.

Expected failures:
  When PACKAGE arguments are given, the script will fail if any requested package is not found in the REPO_ROOT/resources/constraints/*.txt files.

Requirements:
  - The REPO_ROOT must be a valid napari repository with the correct file structure.
  - uv needs to be installed.

Examples:
  compile_constraints.sh
  compile_constraints.sh /path/to/napari
  compile_constraints.sh superqt tifffile
  compile_constraints.sh /path/to/napari superqt tifffile

EOF
}

if [[ "$#" -gt 0 && ("${1:-}" == "-h" || "${1:-}" == "--help") ]]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

repo_root="${SCRIPT_DIR}/.."
if [ "$#" -gt 0 ] && [ -d "$1" ]; then
  repo_root="$1"
  shift
fi

cd "${repo_root}"

# Decide what to pass to uv pip compile: either a global --upgrade (no args)
# or one or more --upgrade-package <pkg> for packages that are present in
# our constraint files under resources/constraints/*.txt
if [ "$#" -eq 0 ]; then
  upgrade_flag=(--upgrade)
else
  # collect constraint files (use nullglob-like handling)
  constraint_files=("resources/constraints"/*.txt)

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
  cat "${constraint_files[@]}" >"$tmp_constraints"

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
uv pip compile --python-version 3.14 --output-file resources/requirements_mypy.txt "${upgrade_flag[@]}" resources/requirements_mypy.in
