# Running napari headlessly

Running napari headlessly (without opening a napari GUI interface) may be
desired for some users, for example to run batch analysis via a CLI.

Headless mode can be achieved by not showing the viewer:

```python
viewer = napari.Viewer(show=False)
# or directly creating a ViewerModel instance
viewer_model = napari.components.ViewerModel()
```

Currently, using `napari.Viewer(show=False)` will *not* prevent
Qt from being imported. This can crash the napari application as on creation
of QApplication, if Qt cannot connect to display it will abort the application.
One way around this is to ensure that QtPy or any of the Qt backends are not
installed. Another option is to set the environment variable:
`QT_QPA_PLATFORM=offscreen` in the environment. This tells Qt backend that
rendering should be done offscreen.

Alternatively, any of the lower level napari components such as `LayerList` and
`Layer` could be directly used, which would not start a napari viewer.
