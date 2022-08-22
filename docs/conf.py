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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from importlib import import_module
from pathlib import Path

import qtgallery
from jinja2.filters import FILTERS

import napari

release = napari.__version__
if "dev" in release:
    version = "dev"
else:
    version = release

# -- Project information -----------------------------------------------------

project = 'napari'
copyright = '2022, The napari team'
author = 'The napari team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
autosummary_generate = True
autosummary_imported_members = True
comments_config = {'hypothesis': False, 'utterances': False}

# execution_allow_errors = False
# execution_excludepatterns = []
# execution_in_temp = False
# execution_timeout = 30

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_external_toc",
    "sphinx_tabs.tabs",
    'myst_nb',
    #    "sphinx_comments",
    "sphinx_panels",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_tags",
]

external_toc_path = "_toc.yml"
external_toc_exclude_missing = False

tags_create_tags = True
tags_output_dir = "_tags"
tags_extension = ["md", "rst"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'napari'

# Define the json_url for our version switcher.
json_url = "https://napari.org/version_switcher.json"

if version == "dev":
    version_match = "latest"
else:
    version_match = release

html_theme_options = {
    "external_links": [
        {"name": "napari hub", "url": "https://napari-hub.org"}
    ],
    "github_url": "https://github.com/napari/napari",
    "navbar_start": ["navbar-project"],
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://napari.org/version_switcher.json",
        "version_match": version_match,
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "images/logo.png"
html_sourcelink_suffix = ''
html_title = 'napari'

html_css_files = [
    'custom.css',
]

intersphinx_mapping = {
    'python': ['https://docs.python.org/3', None],
    'numpy': ['https://numpy.org/doc/stable/', None],
    'napari_plugin_engine': [
        'https://napari-plugin-engine.readthedocs.io/en/latest/',
        'https://napari-plugin-engine.readthedocs.io/en/latest/objects.inv',
    ],
    'magicgui': [
        'https://napari.org/magicgui/',
        'https://napari.org/magicgui/objects.inv',
    ],
}

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'substitution',
    'tasklist',
]

myst_heading_anchors = 3

nb_output_stderr = 'show'

panels_add_bootstrap_css = False
pygments_style = 'solarized-dark'
suppress_warnings = ['myst.header', 'etoc.toctree']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '.jupyter_cache',
    'jupyter_execute',
    'plugins/_*.md',
]

napoleon_custom_sections = [('Events', 'params_style')]


def reset_napari_theme(gallery_conf, fname):
    from napari.settings import get_settings

    settings = get_settings()
    settings.appearance.theme = 'dark'
    qtgallery.reset_qapp(gallery_conf, fname)


sphinx_gallery_conf = {
    'examples_dirs': '../examples',  # path to your example scripts
    'gallery_dirs': 'gallery',  # path to where to save gallery generated output
    'filename_pattern': '/*.py',
    'ignore_pattern': 'README.rst|/*_.py',
    'default_thumb_file': Path(__file__).parent.parent
    / 'napari'
    / 'resources'
    / 'logo.png',
    'plot_gallery': True,
    'download_all_examples': False,
    'min_reported_time': 10,
    'only_warn_on_example_error': True,
    'image_scrapers': (qtgallery.qtscraper,),
    'reset_modules': (reset_napari_theme,),
}


def setup(app):
    """Ignore .ipynb files.

    Prevents sphinx from complaining about multiple files found for document
    when generating the gallery.

    """
    app.registry.source_suffix.pop(".ipynb", None)


def get_attributes(item, obj, modulename):
    """Filters attributes to be used in autosummary.

    Fixes import errors when documenting inherited attributes with autosummary.

    """
    module = import_module(modulename)
    if hasattr(getattr(module, obj), item):
        return f"~{obj}.{item}"
    else:
        return ""


FILTERS["get_attributes"] = get_attributes
