Announcement: napari 0.2.6
==========================

We're happy to announce the release of napari v0.2.6!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
************
- label axes with strings (#644)
- interactive scripting with `viewer.update`(#650)
- add dock widget support (#695)
- dockable console (#714)

Improvements
************
- improve release guide (#668)
- add logo to repo (#674)
- improve labels painting speed (#684)
- add example showing mouse drag callbacks (#690)
- add main window option to screenshots (#722)

Bugfixes
********
- allow all animation thread tests to be +/- 1 frame (#670)
- document qt not qt5 (#677)
- fix init of _position (#680)
- fix 3d display surface (#682)
- set arcballCamera fov default (#683)
- cleaning for interactive scripting (#688)
- change no. of pixels calculation from 32 to 64-bit (#692)
- support multichannel dask array (#701)
- Don't calc_data_range on uint8 data (#705)
- allows Path in io.magic_imread (#709)
- handles empty chosen files and folder (#715)
- relax play_api (#717)
- raise main window when showing (#721)
- fix vertical scrollbars (#728)
- revert "change no. of pixels calculation from 32 to 64-bit" (#738)
- remove vispy backport with 0.6.3, fix segfault in #576 (#739)
- improve pyramid guessing (#740)

7 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Ahmet Can Solak
- Guillaume Gay
- HagaiHargil (HagaiHargil)
- Heath Patterson
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Talley Lambert


5 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- HagaiHargil
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Talley Lambert
