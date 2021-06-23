# napari 0.4.8

We're happy to announce the release of napari 0.4.8!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

This release comes with a **big** change with how you use napari: you should no
longer wrap viewer calls in the `with napari.gui_qt():` context. Instead, when
you want to block and call the viewer, use `napari.run()`. A minimal example:

```python
import napari
from skimage import data

camera = data.camera()
viewer = napari.view_image(camera)
napari.run()
```

In interactive workspaces such as IPython and Jupyter Notebook, you should no
longer need to use `%gui qt`, either: napari will enable it for you.

For more information, examples, and documentation, please visit our website:
https://napari.org

## Highlights

This release adds a new plugin type (i.e. a hook specification) for plugins to
provide sample data (#2483). No more demos with `np.random`! 🎉 We've added a
built-in sample data plugin for this using the scikit-image data module.
Use it with `viewer.open_sample(plugin_name, sample_name)`, for example,
`viewer.open_sample('scikit-image', 'camera')`. Or you can use the File
menu at File -> Open Sample. For more on how to provide your own sample
datasets to napari, see [how to write a
plugin](https://napari.org/plugins/stable/for_plugin_developers.html) and the
[sample data
specification](https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_provide_sample_data).

The scale bar now has rudimentary support for physical units 📏 (#2617). To use
it, set your scale numerically as before, then use `viewer.scale_bar.unit =
'um'`, for example.

We have also added a text overlay, which you can use to display arbitrary text
over the viewer (#2595). You can use this to display time series time stamps,
for example. Access it at `viewer.text_overlay`.

Editing segmentations with napari is easier than ever now with varying number
of dimensions during painting/filling with labels (#2609). Previously, if you
wanted to edit segmentations in a time series, you had to choose between
painting 2D planes, or painting in 4D. Now you can edit individual volumes
without affecting the others.

If you launch a long running process from napari, you can now display a progress
bar on the viewer (#2580). You can find usage examples in the repo
[here](https://github.com/napari/napari/blob/fa342dc399b636330afdb1b4cb58f919832651fd/examples/progress_bar_minimal.py)
and
[here](https://github.com/napari/napari/blob/fa342dc399b636330afdb1b4cb58f919832651fd/examples/progress_bar_segmentation.py).


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
- Add FOV to camera model and slider popup (#2636). Right click on the 2D/3D
  display toggle button to get a perspective projection view in 3D.

## Improvements

- Add stretch to vertical dock widgets (#2154)
- Use new selection model on existing `LayerList` (#2441)
- Add bbox annotator example (#2446)
- Save last working directory (#2467)
- add name to dock widget titlebar (#2471)
- Add Qt`ListModel` and `ListView` for `EventedList` (#2486)
- New new qt layerlist (#2493)
- Move plugin sorter (#2501)
- Add support for passing `shape_type` through `data` attribute for `Shapes` layers (#2507)
- Add ColorManager to Vectors (#2512)
- Enhance translation methods (#2517)
- Cleanup plugins.__init__, better test isolation (#2535)
- Add typing to schema_version (#2536)
- Add initial restart implementation (#2540)
- Add data setter for `surface` layers (#2544)
- Extract shortcut into their own object. (#2554)
- Add example tying slider change to point properties change (#2582)
- Range of label spinbox is more dtype-aware (#2597)
- Add generic name to unnamed dockwidgets (#2604)
- Add option to save state separate from geometry (#2606)
- QtLargeIntSpinbox for label controls (#2608)
- Support varying number of dimensions during labels painting (#2609)
- Add units to the ScaleBar visual (#2617)
- Return widgets created by `add_plugin_dock_widget` (#2635)
- Add _QtMainWindow.current (#2638)
- Relax dask test (#2641)
- Add table header style (#2645)
- QLargeIntSpinbox with QAbstractSpinbox and python model (#2648)
- Add Labels layer `get_dtype` utility to account for multiscale layers (#2679)
- Display file format options when saving layers (#2650)
- Add events to plugin manager (#2663)
- Add napari module to console namespace (#2687)
- Change deprecation warnings to future warnings (#2707)
- Add strict_qt and block_plugin_discovery parameters to make_napari_viewer (#2715)

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
- Make sure Delete is a special key mapping (#2613)
- Disconnect some events on Canvas destruction (#2615)
- Add missing QSpinBox import in Labels layer controls (#2619)
- Use dtype.type when passing to downstream NumPy functions (#2632)
- Fix notifications when something other than napari or ipython creates QApp (#2633)
- Update missing translations for 0.4.8 (#2664)
- Catch dockwidget layout modification error (#2671)
- Fix warnings in thread_worker, relay messages to gui (#2688)
- Add missing setters for shape attributes (#2696)
- Add get_default_shape_type utility introspecting current shape type (#2701)
- Fix handling of exceptions and notifications of threading threads (#2703)
- Fix vertical_stretch injection and kwargs passing on DockWidget (#2705)
- Fix tracks icons, and visibility icons (#2708)
- Patch horizontalAdvance for older Qt versions (#2711)
- Fix segfaults in test (#2716) 
- Fix napari_provide_sample_data documentation typo (#2718)
- Fix mpl colormaps (#2719)
- Fix active layer keybindings (#2722)
- Fix labels with large maximum value (#2723)
- Fix progressbar and notifications segfaults in test (#2726)

## API Changes

- By default, napari used to create a dask cache. This has caused unforeseen
  bugs, though, so it will no longer be done automatically. (#2590) If you
  notice a drop in performance for your dask+napari use case, you can restore
  the previous behaviour with
  `napari.utils.resize_dask_cache(memory_fraction=0.1)`. You can of course also
  experiment with other values!
- The default `area` for `add_dock_widget` is now `right`, and no longer `bottom`.
- To avoid oddly spaced sparse widgets, #2154 adds vertical stretch to the
  bottom of all dock widgets added (via plugins or manually) with an `area`
  of `left` or `right`, *unless:*

    1) the widget, or any widget in its primary layout, has a vertical
       [`QSizePolicy`](https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
       of `Expanding`, `MinimumExpanding`, or `Ignored`

    1) `add_vertical_stretch=False` is provided to `add_dock_widget`,
       or in the widget options provided with plugin dock widgets.


## Deprecations

- As noted at the top of these notes, `napari.gui_qt()` is deprecated (#2533).
  Call `napari.run()` instead when you want to display the napari UI.

## UI changes

- Toggle theme has been removed from the menubar. (#2462) Instead, change the
  theme in the preferences panel.
- The number of 2D interpolation options available from the drop down menu has
  been reduced. (#2552)
- The ipy interactive setting has been removed from the preferences panel.
  (#2605) You can still turn it off from the API with
  `napari.utils.settings.get_settings().ipy_interactive = False`, but this is not
  recommended.
- The `n-dimensional` tick box in the Labels layer controls has been removed.
  (#2609) Use "n edit dims" instead.

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
- Example using matplotlib figure (#2668)

## Build Tools and Support

- Add a simple property-based test using Hypothesis (#2469)
- Add check for strings missing translations (#2521)
- Check if opengl file exists (#2630)
- Remove test warnings again, minimize output, hide more async stuff (#2642)
- Remove `raw_stylesheet` (#2643)
- Add link to top level project roadmap page (#2652)
- Replace pypa/pep517 with pypa/build (#2684)
- Add provide sample data hook to docs (#2689)

## Other Pull Requests

- Update PULL_REQUEST_TEMPLATE.md (#2497)
- Non-dynamic base layer classes (#2624)


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
- [Pam Wadhwa](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
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
