Announcement: napari 0.2.11
===========================

We're happy to announce the release of napari 0.2.11!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

Highlights
**********
- Point face color and edge color are now settable as a property in a columnar data table, mapped using a colormap (continuous values) or a color cycle (categorical values). See `this example <https://github.com/napari/napari/blob/master/examples/add_points_with_properties.py>`_ for syntax details.
- Python 3.8 is now supported.

Improvements
************
- Represent color as Nx4 array in Points  (#782)
- Qt/Vispy connection & lambda refactor (#859)
- Improve slider step precision based on data range (#884)
- Allow Points display properties to be set by point properties (#895)
- Refactor add_* methods (#897)
- Add _add_layer_data method and tests (#909)

Bugfixes
********
- Fix clim popup position in floating widgets (#869)
- Fix points selection (#902)
- Move benchmarks under napari directory but not distributed (#913)
- Patch py3.8 on windows (#915)
- Fix nD Shapes.to_labels (#920)
- Fix singleton dims (#923)

Support
*******
- Add PyQt5 tests to Linux and OSX CI (#867)
- Change autodevdoc script to use new git reference (#876)
- Fix invalid instructions in setup.py (#877)
- Bump tests to python 3.8, general tests fix (#878)
- Clean up setup.py (#880)
- Fix osx py3.6 tests (#916)
- Add Python 3.8 to PyPI tags (#917)
- Relocate existing tests for appropriate discoverability. (#918)
- Import scipy stats to prevent strange bug in tests (#927)

7 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Hagai Har-Gil
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Reece Dunham
- Talley Lambert
- Tony Tung


8 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Hagai Har-Gil
- Juan Nunez-Iglesias
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Reece Dunham
- Talley Lambert
- Tony Tung
