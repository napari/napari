import os.path as osp

import jupytext
from nbconvert.preprocessors import ExecutePreprocessor


examples_folder = osp.join(osp.dirname(osp.dirname(__file__)), 'examples')
ep = ExecutePreprocessor(timeout=1000)


def jupytext_execute(filename, format_name=None, freeze_metadata=False):
    # https://nbconvert.readthedocs.io/en/latest/execute_api.html
    notebook = jupytext.readf(filename, format_name=format_name,
                              freeze_metadata=freeze_metadata)
    return ep.preprocess(notebook,
                         {'metadata': {'path': osp.dirname(filename)}})


def test_layers():
    layers = osp.join(examples_folder, 'layers.md')
    jupytext_execute(layers)

    
def test_markers():
    layers = osp.join(examples_folder, 'markers.md')
    jupytext_execute(layers)
