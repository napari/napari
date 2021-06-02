# napari 0.4.9

We're happy to announce the release of napari 0.4.9!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari


## Highlights
This release adds a couple nice new features like additional shading modes for our
surface layer (#2972) and the ability to copy a screenshot directly to the clipboard (#2721).
It also contains a variety of bug fixes and improvements.


## New Features
- Added copy-to-clipboard functionality (#2721)
- Add `make watch` command for hot reload (#2763)
- Expose alternative shading modes for surfaces (#2792)


## Improvements
- Global plugin setting (#2565)
- Delay all imports in `napari.__init__` behind module level `napari.__getattr__` (#2662)
- Add `block` to viewer.show (#2669)
- New type stubs PR, and simpler `napari.view_layers` module (#2675)
- Add `MultiScaleData` wrapper to give multiscale data a consistent API (#2683)
- Revert "add `MultiScaleData` wrapper to give multiscale data a consistent API (#2683)" (#2807)
- Add repr-html to nbscreenshot (#2740)
- Set default highlight thickness to 2 (#2746)
- Save shortcuts in settings (#2754)
- Enable correct loading of settings from environment variables (#2759)
- Add tooltips to widgets in preferences (#2762)
- Improve colormap error message, when using display names or wrong colormap names (#2769)
- Add parent to console and dockwidgets in a separate private attribute. (#2773)
- Improve error message when legacy Qt installed from conda over pip (#2776)
- Change remove to uninstall in plugin dialog (#2787)
- Do not write settings loaded from environment values (#2797)


## Bug Fixes
- Fixing `add_dock_widget` compatibility with `magicgui v0.2` (#2734)
- Shortcuts: Render properly shortcuts with minus and space. (#2735)
- Fix runtime error when running doc tests on napari site (#2738)
- Fix play button with dask/zarr (#2741)
- Ignore opening 1D image layers in the viewer model. (#2743)
- Fix custom layer subclasses (don't require layer icon) (#2758)
- Add temporal fix for handling enums in qt_json_form (#2761)
- Use slicing rather than np.take in add_image (#2780)
- Fix paint cursor size for layer with negative scale (#2788)
- track_id dtype change. from uint16 to uint32 (#2789)
- Ensure aborted workers don't emit returned signal (#2796)
- Connect axes visual to dims range change (#2802)
- Add fix for large labels in new slices (#2804)
- Fix zoom for non square image (#2805)
- Implement lazy module importing for all public submodules (#2816)


## Tasks
- Add imports linting (#2659)
- Pre-commit update (#2728)
- Remove linenos option from code-blocks and line references (#2739)
- Remove qt from _qt import linter rule (#2774)
- Add PR labeler and update templates (#2775)
- Add pytest-order and move threading tests to the top of the suite (#2779)
- Auto assign PR to author (#2794)


## 12 authors added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 10 reviewers added to this release (alphabetical)

- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

