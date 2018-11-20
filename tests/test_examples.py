import os
import os.path as osp

import pytest

import jupytext
from jupytext import compare
from nbconvert.preprocessors import ExecutePreprocessor


ep = ExecutePreprocessor(timeout=1000)


def jupytext_execute(path, format_name=None, freeze_metadata=False):
    # https://nbconvert.readthedocs.io/en/latest/execute_api.html
    notebook = jupytext.readf(path, format_name=format_name,
                              freeze_metadata=freeze_metadata)
    return ep.preprocess(notebook,
                         {'metadata': {'path': osp.dirname(path)}})


def jupytext_round_trip(path, format_name=None, freeze_metadata=False,
                        update=False, allow_expected_differences=True,
                        stop_on_first_error=True):
    notebook = jupytext.readf(path, format_name=format_name,
                              freeze_metadata=freeze_metadata)
    return compare.test_round_trip_conversion(notebook,
                                              osp.splitext(path)[1],
                                              format_name, update,
                                              allow_expected_differences,
                                              stop_on_first_error)


examples_folder = osp.join(osp.dirname(osp.dirname(__file__)), 'examples')
excludes = ['__init__.py', 'README.md', 'demo.py']
paths = []
for filename in os.listdir(examples_folder):
    if filename not in excludes:
        for ext in jupytext.NOTEBOOK_EXTENSIONS:
            if filename.endswith(ext):
                paths.append(osp.join(examples_folder, filename))


@pytest.mark.parametrize('path', paths)
def test_round_trip(path):
    jupytext_round_trip(path)


@pytest.mark.parametrize('path', paths)
def test_execute(path):
    jupytext_execute(path)
