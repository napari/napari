import os
from os.path import abspath, dirname

from . import qt

resources_dir = abspath(dirname(__file__))

with open(os.path.join(resources_dir, 'stylesheet.qss'), 'r') as f:
    stylesheet = f.read()
