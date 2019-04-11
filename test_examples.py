import os
import os.path as osp

import pytest


os.environ['NAPARI_TEST'] = '1'

root_dir = osp.dirname(__file__)
examples_dir = osp.join(root_dir, 'examples')
excludes = ['__init__.py']

paths = []
for filename in os.listdir(examples_dir):
    if filename not in excludes and osp.splitext(filename)[1] == '.py':
        paths.append(osp.join(examples_dir, filename))


def path_id(name):
    if name.startswith(root_dir):
        return name[len(root_dir) + 1:]
    return name


@pytest.mark.parametrize('path', paths, ids=path_id)
def test_example(path):
    with open(path, 'r') as f:
        print(path)
        exec(f.read(), {})
