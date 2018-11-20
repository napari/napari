import os
import os.path as osp

import pytest

import jupytext
from nbconvert.preprocessors import ExecutePreprocessor


ep = ExecutePreprocessor(timeout=1000)


def jupytext_execute(path, format_name=None, freeze_metadata=False):
    # https://nbconvert.readthedocs.io/en/latest/execute_api.html
    notebook = jupytext.readf(path, format_name=format_name,
                              freeze_metadata=freeze_metadata)
    return ep.preprocess(notebook,
                         {'metadata': {'path': osp.dirname(path)}})


examples_folder = osp.join(osp.dirname(osp.dirname(__file__)), 'examples')
excludes = ['__init__.py', 'README.md', 'demo.py']
filenames = []
for filename in os.listdir(examples_folder):
    if filename not in excludes:
        for ext in jupytext.NOTEBOOK_EXTENSIONS:
            if filename.endswith(ext):
                filenames.append(filename)


@pytest.mark.parametrize('filename', filenames)
def test_execute(filename):
    path = osp.join(examples_folder, filename)
    jupytext_execute(path)
