# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
import fileinput
import recommonmark  # noqa: F401
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('../..'))
from napari import __version__  # noqa: E402


# -- Project information -----------------------------------------------------

project = 'napari'
copyright = '2020, napari contributors'
author = 'napari contributors'

release = __version__
version = __version__
CONFDIR = os.path.dirname(__file__)


def clean_release_notes():

    release_rst = """Release Notes
=============

.. toctree::
   :maxdepth: 1
   :glob:

"""
    dirname = os.path.join(CONFDIR, 'release')
    for rel in sorted(
        os.listdir(dirname),
        key=lambda s: list(map(int, re.findall(r'\d+', s))),
        reverse=True,
    ):
        for line in fileinput.input(os.path.join(dirname, rel), inplace=True):
            line = re.sub(
                r'#(\d+)',
                r'[#\1](<https://github.com/napari/napari/issues/\1>)',
                line,
            )
            print(line, end='')
        release_rst += '   release/' + rel.replace('.md', '\n')
    with open(os.path.join(CONFDIR, 'releases.rst'), 'w') as f:
        f.write(release_rst)


clean_release_notes()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# Custom parsers of source files.
source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

# intersphinx allows us to link directly to other repos sphinxdocs.
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'napari_plugin_engine': (
        'https://napari-plugin-engine.readthedocs.io/en/latest/',
        'https://napari-plugin-engine.readthedocs.io/en/latest/objects.inv',
    ),
    # 'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = "img/napari_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Extension configuration -------------------------------------------------

# add_module_names = False avoids showing the full path to a function or class
# for example:
# napari.layers.points.keybindings.activate_add_mode(layer)
# becomes
# activate_add_mode
# (we can show the full module path elsewhere on the page)

add_module_names = False

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# app setup hook
github_doc_root = 'https://github.com/napari/napari/tree/master/docs/'


def setup(app):
    app.add_config_value(
        'recommonmark_config',
        {
            'url_resolver': lambda url: github_doc_root + url,
            'enable_auto_toc_tree': True,
            'auto_toc_tree_section': 'Contents',
            'enable_eval_rst': True,
        },
        True,
    )
    app.add_transform(AutoStructify)
