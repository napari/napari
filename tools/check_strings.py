"""
Script to check for string in the codebase not using `trans`.

TODO:
  * Add lines where text is found for convenience?
  * Find all logger calls and add to skips?
  * Find nested funcs inside if/else?
"""

import ast
import os
import sys
import tokenize
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from strings_list import (
    SKIP_FILES,
    SKIP_FOLDERS,
    SKIP_WORDS,
    SKIP_WORDS_GLOBAL,
)


class FindTransStrings(ast.NodeVisitor):
    """This node visitor finds translated strings."""

    def __init__(self):
        super().__init__()
        self._found = set()

    def visit_Call(self, node):
        try:
            if node.func.value.id == "trans":
                for item in [arg.value for arg in node.args]:
                    self._found.add(item)
        except Exception:
            pass

        self.generic_visit(node)


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
    def : list of ast.FunctionDef
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
):
    """Find recursively all files in path.

    Parameters
    ----------
    path : str
        Path to a folder to find files in.
    skip_folders : tuple
        Skip folders containing folder to skip
    skip_files : tuple
        Skip files.
    extensions: tuple, optional
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


def find_docstrings(fpath: str):
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
    dict of tuples
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


def find_trans_strings(fpath: str) -> Dict[str, str]:
    """Find all translation strings for the given file.

    Parameters
    ----------
    fpath : str
        File path.

    Returns
    -------
    dict
        A dict with a stripped string as key and the orginal string for value.
    """
    with open(fpath) as fh:
        data = fh.read()

    module = ast.parse(data)
    trans_strings = {}
    show_trans_strings.visit(module)
    for string in show_trans_strings._found:
        key = " ".join([it for it in string.split()])
        trans_strings[key] = string

    show_trans_strings._found = set()

    return trans_strings


def import_module_by_path(fpath: str) -> Optional[ModuleType]:
    """Import a module given py a path.

    Parameters
    ----------
    fpath : str
        The path to the file to import as module.

    Returns
    -------
    ModuleType
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


def find_untranslated_strings(
    paths: List[str], skip_words: List[str]
) -> Dict[str, List[Tuple[int, str]]]:
    """Find strings that have not been translated.

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
    dict of tuples
        List of issues found per path. Each issue is a tuple with line number
        and the untranslated string.
    """
    issues = {}
    for fpath in paths:
        issues[fpath] = []
        strings = find_strings(fpath)
        trans_strings = find_trans_strings(fpath)
        doc_strings = find_docstrings(fpath)

        skip_words_for_file = skip_words.get(fpath, [])
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

        try:
            _content = (
                "    "
                + repr(fpath)
                + ": "
                + repr(sorted({it[-1] for it in issues[fpath]}))
                + ","
            )
            print(_content)
        except Exception:
            pass

        if not issues[fpath]:
            issues.pop(fpath)

    return issues


if __name__ == "__main__":
    path = sys.argv[1]
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = find_files(path, SKIP_FOLDERS, SKIP_FILES)

    issues = find_untranslated_strings(paths, SKIP_WORDS)
    print("\n\n")
    if issues:
        print(
            "Some strings on the following files might need to be translated "
            "or added to the skip list on the `tools/strings_list.py` "
            "file.\n\n"
        )
        for fpath, values in issues.items():
            print(fpath)
            print(values)
            print("\n")

        sys.exit(1)
    else:
        print("All strings seem to be using translations. Good job!\n\n")
