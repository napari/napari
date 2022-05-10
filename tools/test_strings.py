"""
Script to check for string in the codebase not using `trans`.

TODO:
  * Find all logger calls and add to skips
  * Find nested funcs inside if/else
"""

import ast
import os
import tokenize
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Set, Tuple

import pytest
from strings_list import (
    SKIP_FILES,
    SKIP_FOLDERS,
    SKIP_WORDS,
    SKIP_WORDS_GLOBAL,
)

REPO_ROOT = Path(__file__).resolve()
NAPARI_MODULE = (REPO_ROOT / "napari").relative_to(REPO_ROOT)

# Types
StringIssuesDict = Dict[str, List[Tuple[int, str]]]
OutdatedStringsDict = Dict[str, List[str]]
TranslationErrorsDict = Dict[str, List[Tuple[str, str]]]


class FindTransStrings(ast.NodeVisitor):
    """This node visitor finds translated strings."""

    def __init__(self):
        super().__init__()

        self._found = set()
        self._trans_errors = []

    def _check_vars(self, method_name, args, kwargs):
        """Find interpolation variables inside a translation string.

        This helps find any variables that need to be interpolated inside
        a string so we can check against the `kwargs` for both singular
        and plural strings (if present) inside `args`.

        Parameters
        ----------
        method_name : str
            Translation method used. Options include "_", "_n", "_p" and
            "_np".
        args : list
            List of arguments passed to translation method.
        kwargs : kwargs
            List of keyword arguments passed to translation method.
        """
        singular_kwargs = set(kwargs) - set({"n"})
        plural_kwargs = set(kwargs)

        # If using trans methods with `context`, remove it since we are
        # only interested in the singular and plural strings (if any)
        if method_name in ["_p", "_np"]:
            args = args[1:]

        # Iterate on strings passed to the trans method. Could be just a
        # singular string or a singular and a plural. We use the index to
        # determine which one is used.
        for idx, arg in enumerate(args):
            found_vars = set()
            check_arg = arg[:]
            check_kwargs = {}
            while True:
                try:
                    check_arg.format(**check_kwargs)
                except KeyError as err:
                    key = err.args[0]
                    found_vars.add(key)
                    check_kwargs[key] = 0
                    continue

                break

            if idx == 0:
                check_1 = singular_kwargs - found_vars
                check_2 = found_vars - singular_kwargs
            else:
                check_1 = plural_kwargs - found_vars
                check_2 = found_vars - plural_kwargs

            if check_1 or check_2:
                error = (arg, check_1.union(check_2))
                self._trans_errors.append(error)

    def visit_Call(self, node):
        method_name, args, kwargs = "", [], []
        try:
            if node.func.value.id == "trans":
                method_name = node.func.attr

                # Args
                args = []
                for item in [arg.value for arg in node.args]:
                    args.append(item)
                    self._found.add(item)

                # Kwargs
                kwargs = []
                for item in [kw.arg for kw in node.keywords]:
                    if item != "deferred":
                        kwargs.append(item)

        except Exception:
            pass

        if method_name:
            self._check_vars(method_name, args, kwargs)

        self.generic_visit(node)

    def reset(self):
        """Reset variables storing found strings and translation errors."""
        self._found = set()
        self._trans_errors = []


show_trans_strings = FindTransStrings()


def _find_func_definitions(
    node: ast.AST, defs: List[ast.FunctionDef] = []
) -> List[ast.FunctionDef]:
    """Find all functions definition recrusively.

    This also find functions nested inside other functions.

    Parameters
    ----------
    node : ast.Node
        The initial node of the ast.
    defs : list of ast.FunctionDef
        A list of function definitions to accumulate.

    Returns
    -------
    list of ast.FunctionDef
        Function definitions found in `node`.
    """
    try:
        body = node.body
    except Exception:
        body = []

    for node in body:
        _find_func_definitions(node, defs=defs)
        if isinstance(node, ast.FunctionDef):
            defs.append(node)

    return defs


def find_files(
    path: str,
    skip_folders: tuple,
    skip_files: tuple,
    extensions: tuple = (".py",),
) -> List[str]:
    """Find recursively all files in path.

    Parameters
    ----------
    path : str
        Path to a folder to find files in.
    skip_folders : tuple
        Skip folders containing folder to skip
    skip_files : tuple
        Skip files.
    extensions : tuple, optional
        Extensions to filter by. Default is (".py", )

    Returns
    -------
    list
        Sorted list of found files.
    """
    found_files = []
    for root, _dirs, files in os.walk(path, topdown=False):
        for filename in files:
            fpath = os.path.join(root, filename)

            if any(folder in fpath for folder in skip_folders):
                continue

            if fpath in skip_files:
                continue

            if filename.endswith(extensions):
                found_files.append(fpath)

    return list(sorted(found_files))


def find_docstrings(fpath: str) -> Dict[str, str]:
    """Find all docstrings in file path.

    Parameters
    ----------
    fpath : str
        File path.

    Returns
    -------
    dict
        Simplified string as keys and the value is the original docstring
        found.
    """
    with open(fpath) as fh:
        data = fh.read()

    module = ast.parse(data)
    docstrings = []
    function_definitions = _find_func_definitions(module)
    docstrings.extend([ast.get_docstring(f) for f in function_definitions])
    class_definitions = [
        node for node in module.body if isinstance(node, ast.ClassDef)
    ]
    docstrings.extend([ast.get_docstring(f) for f in class_definitions])
    method_definitions = []

    for class_def in class_definitions:
        method_definitions.extend(
            [
                node
                for node in class_def.body
                if isinstance(node, ast.FunctionDef)
            ]
        )

    docstrings.extend([ast.get_docstring(f) for f in method_definitions])
    docstrings.append(ast.get_docstring(module))
    docstrings = [doc for doc in docstrings if doc]

    results = {}
    for doc in docstrings:
        key = " ".join([it for it in doc.split() if it != ""])
        results[key] = doc

    return results


def find_strings(fpath: str) -> Dict[Tuple[int, str], Tuple[int, str]]:
    """Find all strings (and f-strings) for the given file.

    Parameters
    ----------
    fpath : str
        File path.

    Returns
    -------
    dict
        A dict with a tuple for key and a tuple for value. The tuple contains
        the line number and the stripped string. The value containes the line
        number and the original string.
    """
    strings = {}
    with open(fpath) as f:
        for toktype, tokstr, (lineno, _), _, _ in tokenize.generate_tokens(
            f.readline
        ):
            if toktype == tokenize.STRING:
                try:
                    string = eval(tokstr)
                except Exception:
                    string = eval(tokstr[1:])

                if isinstance(string, str):
                    key = " ".join([it for it in string.split() if it != ""])
                    strings[(lineno, key)] = (lineno, string)

    return strings


def find_trans_strings(
    fpath: str,
) -> Tuple[Dict[str, str], List[Tuple[str, Set[str]]]]:
    """Find all translation strings for the given file.

    Parameters
    ----------
    fpath : str
        File path.

    Returns
    -------
    tuple
        The first item is a dict with a stripped string as key and the
        orginal string for value. The second item is a list of tuples that
        includes errors in translations.
    """
    with open(fpath) as fh:
        data = fh.read()

    module = ast.parse(data)
    trans_strings = {}
    show_trans_strings.visit(module)
    for string in show_trans_strings._found:
        key = " ".join([it for it in string.split()])
        trans_strings[key] = string

    errors = list(show_trans_strings._trans_errors)
    show_trans_strings.reset()
    return trans_strings, errors


def import_module_by_path(fpath: str) -> Optional[ModuleType]:
    """Import a module given py a path.

    Parameters
    ----------
    fpath : str
        The path to the file to import as module.

    Returns
    -------
    ModuleType or None
        The imported module or `None`.
    """
    import importlib.util

    fpath = fpath.replace("\\", "/")
    module_name = fpath.replace(".py", "").replace("/", ".")
    try:
        spec = importlib.util.spec_from_file_location(module_name, fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        module = None

    return module


def find_issues(
    paths: List[str], skip_words: List[str]
) -> Tuple[StringIssuesDict, OutdatedStringsDict, TranslationErrorsDict]:
    """Find strings that have not been translated, and errors in translations.

    This will not raise errors but return a list with found issues wo they
    can be fixed at once.

    Parameters
    ----------
    paths : list of str
        List of paths to files to check.
    skip_words : list of str
        List of words that should be skipped inside the given file.

    Returns
    -------
    tuple
        The first item is a dictionary of the list of issues found per path.
        Each issue is a tuple with line number and the untranslated string.
        The second item is a dictionary of files that contain outdated
        skipped strings. The third item is a dictionary of the translation
        errors found per path. Translation errors referes to missing
        interpolation variables, or spelling errors of the `deferred` keyword.
    """
    issues = {}
    outdated_strings = {}
    trans_errors = {}
    for fpath in paths:
        issues[fpath] = []
        strings = find_strings(fpath)
        trans_strings, errors = find_trans_strings(fpath)
        doc_strings = find_docstrings(fpath)

        skip_words_for_file = skip_words.get(fpath, [])
        skip_words_for_file_check = skip_words_for_file[:]
        module = import_module_by_path(fpath)
        try:
            __all__strings = module.__all__
        except Exception:
            __all__strings = []

        for key in strings:
            _lineno, string = key
            _lineno, value = strings[key]

            if (
                string not in doc_strings
                and string not in trans_strings
                and value not in skip_words_for_file
                and value not in __all__strings
                and string != ""
                and string.strip() != ""
                and value not in SKIP_WORDS_GLOBAL
            ):
                issues[fpath].append((_lineno, value))
            elif value in skip_words_for_file_check:
                skip_words_for_file_check.remove(value)

        if skip_words_for_file_check:
            outdated_strings[fpath] = skip_words_for_file_check

        if errors:
            trans_errors[fpath] = errors

        if not issues[fpath]:
            issues.pop(fpath)

    return issues, outdated_strings, trans_errors


# --- Fixture
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def checks():
    paths = find_files(NAPARI_MODULE, SKIP_FOLDERS, SKIP_FILES)
    issues, outdated_strings, trans_errors = find_issues(paths, SKIP_WORDS)
    return issues, outdated_strings, trans_errors


# --- Tests
# ----------------------------------------------------------------------------
def test_missing_translations(checks):
    issues, _, _ = checks
    print(
        "\nSome strings on the following files might need to be translated "
        "or added to the skip list.\nSkip list is located at "
        "`tools/strings_list.py` file.\n\n"
    )
    for fpath, values in issues.items():
        print(f"{fpath}\n{'*' * len(fpath)}")
        unique_values = set()
        for line, value in values:
            unique_values.add(value)
            print(f"{line}:\t{repr(value)}")

        print("\n")

        if fpath in SKIP_WORDS:
            print(
                f"List below can be copied directly to `tools/strings_list.py` file inside the '{fpath}' key:\n"
            )
            for value in sorted(unique_values):
                print(f"        {repr(value)},")
        else:
            print(
                "List below can be copied directly to `tools/strings_list.py` file:\n"
            )
            print(f"    {repr(fpath)}: [")
            for value in sorted(unique_values):
                print(f"        {repr(value)},")
            print("    ],")

        print("\n")

    no_issues = not issues
    assert no_issues


def test_outdated_string_skips(checks):
    _, outdated_strings, _ = checks
    print(
        "\nSome strings on the skip list on the `tools/strings_list.py` are "
        "outdated.\nPlease remove them from the skip list.\n\n"
    )
    for fpath, values in outdated_strings.items():
        print(f"{fpath}\n{'*' * len(fpath)}")
        print(", ".join(repr(value) for value in values))
        print("")

    no_outdated_strings = not outdated_strings
    assert no_outdated_strings


def test_translation_errors(checks):
    _, _, trans_errors = checks
    print(
        "\nThe following translation strings do not provide some "
        "interpolation variables:\n\n"
    )
    for fpath, errors in trans_errors.items():
        print(f"{fpath}\n{'*' * len(fpath)}")
        for string, variables in errors:
            print(f"String:\t\t{repr(string)}")
            print(
                f"Variables:\t{', '.join(repr(value) for value in variables)}"
            )
            print("")

        print("")

    no_trans_errors = not trans_errors
    assert no_trans_errors
