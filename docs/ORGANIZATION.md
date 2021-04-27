# Organization of Documentation for napari

The organization of documentation for the napari project is currently split into three repositories:
the [`napari/napari`] repo which houses the source code for napari, the [`napari/docs`] repo which contains legacy and version-specific API documentation and guides, and the [`napari/napari.github.io`] repo that makes up the main [napari.org] site, comprising of tutorials and copied-over files from the [`napari/napari`] repo.

## Location of documentation sources

API docs, guides, plugins, roadmaps, releases, developer guides, developer resources, and community resources live in the [`napari/napari`] repo under the `docs` directory. For backwards compatibility with the previous structure as used in [`napari/docs`], some files are "duplicated" by using the [MyST `include` directive](https://myst-parser.readthedocs.io/en/latest/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).

Tutorials, the main index page, and the WIP sphinx theme are in the [`napari/napari.github.io`] repo.

The [`napari/docs`] repo contains no original documentation sources.

## Bringing it all together

For the [`napari/docs`] repo, files are built in `docs/_build/html` in the main [`napari/napari`] repo and then copied over to the appropriate version number or `dev` using an automated process.

Documentation sources are copied over to `napari.github.io` using the `copy-docs.py` script (found in the `docs` directory of [`napari/napari`]). Duplicate files are excluded and the table of contents is automatically updated. Specifying which files should be copied / excluded can be modified in the script itself. This script is automatically run in continuous integration and the changes are pushed to the [`napari/napari.github.io`] repo, which will then automatically build and update the site.

[`napari/napari`]: https://github.com/napari/napari
[`napari/docs`]: https://github.com/napari/docs
[`napari/napari.github.io`]: https://github.com/napari/napari.github.io
[napari.org]: https://napari.org
