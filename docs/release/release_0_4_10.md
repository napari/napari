# napari 0.4.10

We're happy to announce the release of napari 0.4.10!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This is a fairly small release, that follows on quickly from 0.4.9 to fix a regression in
our ability to save layer data (fixed in #2876). It also contains some improvements to our
progress bars (in #2654) and how we compose affine and scale/translate transforms on the
layers (in #2855).


## Improvements
- Add nesting support for progress bars (#2654)
- Auto generate documentation for preferences (#2672)
- Add support for setting the settings configuration path via CLI and import (#2760)
- Minimal changes to support affine composition (#2855)
- Add ActionManager tests (#2868)
- Include the message in notification REPR. (#2874)
- Make `EventedModel` compatible with `dask.Delayed` objects (#2879)
- Add more shortcuts to settings (#2882)


## Bug Fixes
- Typo in action_manager.py (#2869)
- Tifffile compress' kwargs deprecated. Update to compression. (#2872)
- Fix save and update tests (#2876)
- Fix not saving values in settings when loaded from env variables (#2877)
- Fix ipython + visible console results in AttributeError (#2881)
- Fix for too-late magicgui type registration #2891

## API Changes
- In #2855 we have now changed the composition behavior of the affine kwarg and the individual
scale, translate, rotate, and shear kwargs on the layers. Before this release if affine was passed
the others would be ignored. Now they will be composed as `affine * (rotate * shear * scale + translate)`.


## Build Tools
- Call import-linter only in CI (#2878)


## 8 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 10 reviewers added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
