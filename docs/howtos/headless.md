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
One around this is to ensure that QtPy and any of the Qt components is not
installed. Another option is to set the environment variable:
`QT_QPA_PLATFORM=offscreen in the environment`. This tells Qt backend that
rendering should be done offscreen.
