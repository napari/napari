# napari 0.3.1

We're happy to announce the release of napari 0.3.1!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

This is a bug fix release to address issues that snuck through into 0.3.0.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari


## Improvements
- CLI accepts --plugin or any add_* kwargs (#1220)
- Specify viewer.open(plugins='builtins') for all tests (#1222)
- Unify user/plugin kwargs.  Use filename for layer name (#1232)

## Bug Fixes
- rework dask cache (#1206)
- Use grayscale when n_channels=1 (#1217)
- Better error on magic_imread with no files (#1218)
- Improve plugin error messages, bump napari-plugin-engine (#1219)
- make skimage data fixtures compatible with 0.17.0 (#1223)
- Better icon-building strategy (#1229)
- Unpin Jupyter client, issue seems to have resolved (#1240)
- Don't try to get an event.key name if there is no event.key (#1241)
- Update guess_multiscale to deal with strange inputs (#1244)


## Support
- Don't build wheels with releases (#1215)
- Update github issues templates with links to image.sc and zulip  (#1234)
- add new performance doc in new "explanations" directory (#1239)


## 3 authors added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 5 reviewers added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
