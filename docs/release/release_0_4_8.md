# napari 0.4.8

We're happy to announce the release of napari 0.4.8!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

This release comes with a *big* change with how you use napari: you should no
longer wrap viewer calls in the `with napari.gui_qt():` context. Instead, when
you want to block and call the viewer, use `napari.run()`. A minimal example:

```python
import napari
from skimage import data

camera = data.camera()
viewer = napari.view_image(camera)
napari.run()
```

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

- Add Hookspec for providing sample data (#2483)
- Text overlay visual (#2595)
- Support varying number of dimensions during labels painting (#2609)

## New Features

- Add highlight widget to preferences dialog (#2435)
- Add interface language selection to preferences (#2466)
- Add Hookspec for providing sample data (#2483)
- Add ability to run file as plugin (#2503)
- Add `layer.source` attribute to track layer provenance (#2518)
- Add button to drop into debugger in the traceback viewer. (#2534)
- Add initial welcome screen on canvas (#2542)
- Text overlay visual (#2595)
- Add global progress wrapper and ProgressBar widget (#2580)
- Add FOV to camera model and slider popup (#2636)

## Improvements

- Add stretch to vertical dock widgets (#2154)
- Use new selection model on existing `LayerList` (#2441)
- Add bbox annotator example (#2446)
- Save last working directory (#2467)
- add name to dock widget titlebar (#2471)
- Add Qt`ListModel` and `ListView` for `EventedList` (#2486)
- New new qt layerlist (#2493)
- Move plugin sorter (#2501)
- Add ColorManager to Vectors (#2512)
- Enhance translation methods (#2517)
- Cleanup plugins.__init__, better test isolation (#2535)
- Add typing to schema_version (#2536)
- Add initial restart implementation (#2540)
- Add data setter for `surface` layers (#2544)
- Fewer interpolation options (#2552)
- Extract shortcut into their own object. (#2554)
- Add example tying slider change to point properties change (#2582)
- Range of label spinbox is more dtype-aware (#2597)
- Add generic name to unnamed dockwidgets (#2604)
- Hide ipy interactive option (#2605)
- Add option to save state separate from geometry (#2606)
- QtLargeIntSpinbox for label controls (#2608)
- Support varying number of dimensions during labels painting (#2609)
- Return widgets created by `add_plugin_dock_widget` (#2635)
- Add _QtMainWindow.current (#2638)
- Relax dask test (#2641)
- Add table header style (#2645)
- QLargeIntSpinbox with QAbstractSpinbox and python model (#2648)

## Bug Fixes

- Ensure Preferences dialog can only be opened once (#2457)
- Restore QtNDisplayButton (#2464)
- Fix label properties setter (issue #2477) (#2478)
- Fix labels data setter (#2496)
- Fix localization for colormaps (#2498)
- Small brackets fix for Perfmon (#2499)
- Add try except on safe load (#2505)
- Be cautious when checking a module's __package__ attribute (#2516)
- Trigger label colormap generation on seed change to fix shuffle bug, addresses #2523 (#2524)
- Modified quaternion2euler function to cap the arcsin's argument by +-1 (#2530)
- Single line change to track recoloring function (#2532)
- Handle Escape on Preferences dialog (#2537)
- Fix close window handling for non-modals (#2538)
- Fix trans to use new API (#2539)
- Fix set_call_order with missing plugin (#2543)
- Update conditional to use new selection property (#2557)
- Fix visibility toggle (and other key events) in new qt layerlist (#2561)
- Delay importing plugins during settings registration (#2575)
- Don't create a dask cache if it doesn't exist (#2590)
- Update model and actions on menu (#2602)
- Fix z-index of notifications (hidden by welcome window) (#2611)
- Add missing QSpinBox import in Labels layer controls (#2619)
- Use dtype.type when passing to downstream NumPy functions (#2632)
- Fix notifications when something other than napari or ipython creates QApp (#2633)

## API Changes

- Remove toggle theme from drop down menu.  (#2462)
- Add support for passing `shape_type` through `data` attribute for `Shapes` layers (#2507)
- Deprecate gui qt (#2533)
- Don't create a dask cache if it doesn't exist (#2590)
- Non-dynamic base layer classes (#2624)

## Deprecations

- Deprecate gui qt (#2533)
  
## Documentation

- Extend release notes: Add breaking API changes in 0.4.7 (#2494)
- Add about team page (#2508)
- Update translations guide (#2510)
- Misc Doc fixes. (#2515)
- Correct lenght for title underline. (#2541)
- Minor reformatting. (#2555)
- Automate doc copy (#2562)
- Pin docs dependencies (#2568)
- Example of using magicgui with thread_worker (#2577)
- Fix typo in docs CI (#2588)
- Only copy the autosummary templates (#2600)
- Documentation typos (#2614)
- Update event loop documentation for gui_qt deprecation (#2639)

## Build Tools and Support

- Add a simple property-based test using Hypothesis (#2469)
- Add check for strings missing translations (#2521)
- Check if opengl file exists (#2630)
- Remove test warnings again, minimize output, hide more async stuff (#2642)
- Remove `raw_stylesheet` (#2643)
- Add link to top level project roadmap page (#2652)

## Other Pull Requests

- Update PULL_REQUEST_TEMPLATE.md (#2497)


## 19 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Chris Barnes](https://github.com/napari/napari/commits?author=clbarnes) - @clbarnes
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Fifourche](https://github.com/napari/napari/commits?author=Fifourche) - @Fifourche
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Wilson Adhikari](https://github.com/napari/napari/commits?author=wadhikar) - @wadhikar
- [Zac Hatfield-Dodds](https://github.com/napari/napari/commits?author=Zac-HD) - @Zac-HD


## 20 reviewers added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Chris Barnes](https://github.com/napari/napari/commits?author=clbarnes) - @clbarnes
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Emil Melnikov](https://github.com/napari/napari/commits?author=emilmelnikov) - @emilmelnikov
- [Fifourche](https://github.com/napari/napari/commits?author=Fifourche) - @Fifourche
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

