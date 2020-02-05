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
import fileinput

sys.path.insert(0, os.path.abspath('../..'))
from napari import __version__  # noqa: E402


# -- Project information -----------------------------------------------------

project = 'napari'
copyright = '2020, napari contributors'
author = 'napari contributors'

release = __version__
version = __version__


def clean_release_notes():
    dirname = os.path.join(os.path.dirname(__file__), 'release')
    for rel in os.listdir(dirname):
        for line in fileinput.input(os.path.join(dirname, rel), inplace=True):
            if line.startswith("Announcement: napari"):
                line = line.replace("Announcement: ", "")
            # uncomment to remove the standard announcement paragraph.
            # if not line.startswith(
            #     (
            #         "We're happy",
            #         'napari is a fast',
            #         "It's designed for",
            #         "images. It's built",
            #         "rendering), and ",
            #     )
            # ):
            print(line, end='')


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
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


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

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
