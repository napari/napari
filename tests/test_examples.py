import os
import os.path as osp

import pytest

import jupytext
import jupytext.compare
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.fixture(scope='module')
def ep():
    return ExecutePreprocessor(timeout=1000)


root_dir = osp.dirname(osp.dirname(__file__))
examples_dir = osp.join(root_dir, 'examples')
excludes = ['__init__.py', 'README.md', 'demo.py']

paths = []
for filename in os.listdir(examples_dir):
    if filename not in excludes:
        for ext in jupytext.NOTEBOOK_EXTENSIONS:
            if filename.endswith(ext):
                paths.append(osp.join(examples_dir, filename))


def path_id(name):
    if name.startswith(root_dir):
        return name[len(root_dir) + 1:]
    return name


@pytest.mark.parametrize('path', paths,
                         ids=path_id)
def test_round_trip(path):
    ext = osp.splitext(path)[1]
    with open(path) as f:
        notebook = jupytext.reads(path, ext)
    
    jupytext.compare.test_round_trip_conversion(notebook, ext,
                                                format_name=None,
                                                update=False)


@pytest.mark.parametrize('path', paths,
                         ids=path_id)
def test_execute(path, ep):
    # https://nbconvert.readthedocs.io/en/latest/execute_api.html
    notebook = jupytext.readf(path)
    ep.preprocess(notebook,
                  {'metadata': {'path': osp.dirname(path)}})
