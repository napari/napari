Announcement: napari 0.2.12
===========================

We're happy to announce the release of napari 0.2.12!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
************
- Add clickable color swatch with QColorDialog (#832)
- Implement pluggy as plugin manager (#908)
- Allow toggle theme from GUI (#943)
- Add "Screenshot" option to File menu (#944)

Improvements
************
- Allow 3D point selection via API (#907)
- Add show param to Viewer (#961)
- Make mouse drag attributes private in qt_layerlist (#974)
- Rename `viewer` attribute on QtViewerDockWidget to `qt_viewer`(#975)
- Allow LayerList to be initialized by passing in a list of layers (#979)
- Rename `object` to `item` in list input arguments (#980)
- Add layers.events.changed event (#982)

Bugfixes
********
- Fix add points to empty layer (#933)
- Fix points thumbnail (#934)
- Fix 4D ellipses (#950)
- Fix Points highlight bug index (Points-data6-3 test) (#972)
- Fix labels colormap by updating computation of low discrepancy images (#985)
- Pin jupyter-client<6.0.0 (#997)

Support
*******
- Sphinx docs restructuring (create space for dev-focused prose) (#942)
- Fixes broken tutorials links (#946)
- Remove 'in-depth' descriptor for tutorials (#949)
- Flatten auto-generated docs repo (#954)
- Fix bash codeblock in README.md (#956)
- Abstract the construction of view/viewermodel (#957)
- Add docstrings to qt_layerlist.py (#958)
- Fix docs url (#960)
- Fix docs script (#963)
- Fix docs version tag (#964)
- Disallow sphinx 2.4.0; bug fixed in 2.4.1 (#965)
- Remove duplicated imports in setup.py (#969)
- Fix viewer view_* func signature parity (#976)
- Fix ability to test released distributions (#1002)
- Fix recursive-include in manifest.in (#1003)

11 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------
- Genevieve Buckley
- Hagai Har-Gil
- Juan Nunez-Iglesias
- Justin Kiggins
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Peter Boone
- Talley Lambert
- Tony Tung
- Trevor Manz


10 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------
- Ahmet Can Solak
- Clinton Roy
- Genevieve Buckley
- Juan Nunez-Iglesias
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Peter Boone
- Talley Lambert
- Tony Tung
