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

import re
from importlib import import_module
from pathlib import Path
from urllib.parse import urlparse, urlunparse

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
    "sphinx-favicon",
    "sphinx_gallery.gen_gallery",
    "sphinx_tags",
]

external_toc_path = "_toc.yml"
external_toc_exclude_missing = False

tags_create_tags = True
tags_output_dir = "_tags"
tags_overview_title = "Tags"
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

favicons = [
    {
        # the SVG is the "best" and contains code to detect OS light/dark mode
        "static-file": "favicon/logo-silhouette-dark-light.svg",
        "type": "image/svg+xml",
    },
    {
        # Safari in Oct. 2022 does not support SVG
        # an ICO would work as well, but PNG should be just as good
        # setting sizes="any" is needed for Chrome to prefer the SVG
        "sizes": "any",
        "static-file": "favicon/logo-silhouette-192.png",
    },
    {
        # this is used on iPad/iPhone for "Save to Home Screen"
        # apparently some other apps use it as well
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "static-file": "favicon/logo-noborder-180.png",
    },
]

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
    'reference_url': {'napari': None},
}


def setup(app):
    """Ignore .ipynb files.

    Prevents sphinx from complaining about multiple files found for document
    when generating the gallery.

    """
    app.registry.source_suffix.pop(".ipynb", None)
    app.connect('linkcheck-process-uri', rewrite_github_anchor)


def get_attributes(item, obj, modulename):
    """Filters attributes to be used in autosummary.

    Fixes import errors when documenting inherited attributes with autosummary.

    """
    module = import_module(modulename)
    if hasattr(module, "__all__") and obj not in module.__all__:
        return ""

    if hasattr(getattr(module, obj), item):
        return f"~{obj}.{item}"
    else:
        return ""


FILTERS["get_attributes"] = get_attributes

autosummary_ignore_module_all = False

linkcheck_anchors_ignore = [r'^!', r'L\d+-L\d+', r'r\d+', r'issuecomment-\d+']

linkcheck_ignore = ['https://napari.zulipchat.com/']


def rewrite_github_anchor(app, uri: str):
    """Rewrite anchor name of the hyperlink to github.com

    The hyperlink anchors in github.com are dynamically generated.  This rewrites
    them before checking and makes them comparable.
    """
    parsed = urlparse(uri)
    if parsed.hostname == "github.com" and parsed.fragment:
        for text in [
            "L",
            "readme",
            "pullrequestreview",
            "issuecomment",
            "issue",
        ]:
            if parsed.fragment.startswith(text):
                return None
        if re.match(r'r\d+', parsed.fragment):
            return None
        prefixed = parsed.fragment.startswith('user-content-')
        if not prefixed:
            fragment = f'user-content-{parsed.fragment}'
            return urlunparse(parsed._replace(fragment=fragment))
    return None
