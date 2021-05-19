# Events Reference


## Viewer Events

<span style="font-size:0.8em;">
<!-- VIEWER EVENTS TABLE -->
                                                                                                                                                            
|  Class          |  Event Name      |  From viewer                             |  Emitted when ___ changes        |  Event Attribute(s)                   |
|-----------------|------------------|------------------------------------------|----------------------------------|---------------------------------------|
|  `Axes`         |  `visible`       |  `viewer.axes.events.visible`            |                                  |  value: `bool`                        |
|  `Axes`         |  `labels`        |  `viewer.axes.events.labels`             |                                  |  value: `bool`                        |
|  `Axes`         |  `colored`       |  `viewer.axes.events.colored`            |                                  |  value: `bool`                        |
|  `Axes`         |  `dashed`        |  `viewer.axes.events.dashed`             |                                  |  value: `bool`                        |
|  `Axes`         |  `arrows`        |  `viewer.axes.events.arrows`             |                                  |  value: `bool`                        |
|  `Camera`       |  `center`        |  `viewer.camera.events.center`           |                                  |  value: `Tuple[float, float, float]`  |
|  `Camera`       |  `zoom`          |  `viewer.camera.events.zoom`             |                                  |  value: `float`                       |
|  `Camera`       |  `angles`        |  `viewer.camera.events.angles`           |                                  |  value: `Tuple[float, float, float]`  |
|  `Camera`       |  `perspective`   |  `viewer.camera.events.perspective`      |                                  |  value: `float`                       |
|  `Camera`       |  `interactive`   |  `viewer.camera.events.interactive`      |                                  |  value: `bool`                        |
|  `Cursor`       |  `position`      |  `viewer.cursor.events.position`         |                                  |  value: `float`                       |
|  `Cursor`       |  `scaled`        |  `viewer.cursor.events.scaled`           |                                  |  value: `bool`                        |
|  `Cursor`       |  `size`          |  `viewer.cursor.events.size`             |                                  |  value: `int`                         |
|  `Cursor`       |  `style`         |  `viewer.cursor.events.style`            |                                  |  value: `CursorStyle`                 |
|  `Dims`         |  `ndim`          |  `viewer.dims.events.ndim`               |  number of dimensions            |  value: `int`                         |
|  `Dims`         |  `ndisplay`      |  `viewer.dims.events.ndisplay`           |  number of displayed dimensions  |  value: `Literal[2, 3]`               |
|  `Dims`         |  `last_used`     |  `viewer.dims.events.last_used`          |  last-used dimension             |  value: `int`                         |
|  `Dims`         |  `range`         |  `viewer.dims.events.range`              |  range in each dimension         |  value: `Tuple[float, float, float]`  |
|  `Dims`         |  `current_step`  |  `viewer.dims.events.current_step`       |  current position                |  value: `int`                         |
|  `Dims`         |  `order`         |  `viewer.dims.events.order`              |  dimension order                 |  value: `int`                         |
|  `Dims`         |  `axis_labels`   |  `viewer.dims.events.axis_labels`        |  dimension labels                |  value: `str`                         |
|  `GridCanvas`   |  `enabled`       |  `viewer.grid.events.enabled`            |                                  |  value: `bool`                        |
|  `GridCanvas`   |  `stride`        |  `viewer.grid.events.stride`             |                                  |  value: `int`                         |
|  `GridCanvas`   |  `shape`         |  `viewer.grid.events.shape`              |                                  |  value: `Tuple[int, int]`             |
|  `ScaleBar`     |  `visible`       |  `viewer.scale_bar.events.visible`       |                                  |  value: `bool`                        |
|  `ScaleBar`     |  `colored`       |  `viewer.scale_bar.events.colored`       |                                  |  value: `bool`                        |
|  `ScaleBar`     |  `ticks`         |  `viewer.scale_bar.events.ticks`         |                                  |  value: `bool`                        |
|  `ScaleBar`     |  `position`      |  `viewer.scale_bar.events.position`      |                                  |  value: `Position`                    |
|  `ScaleBar`     |  `font_size`     |  `viewer.scale_bar.events.font_size`     |                                  |  value: `float`                       |
|  `ScaleBar`     |  `unit`          |  `viewer.scale_bar.events.unit`          |                                  |  value: `str`                         |
|  `TextOverlay`  |  `visible`       |  `viewer.text_overlay.events.visible`    |                                  |  value: `bool`                        |
|  `TextOverlay`  |  `color`         |  `viewer.text_overlay.events.color`      |                                  |  value: `Array`                       |
|  `TextOverlay`  |  `font_size`     |  `viewer.text_overlay.events.font_size`  |                                  |  value: `float`                       |
|  `TextOverlay`  |  `position`      |  `viewer.text_overlay.events.position`   |                                  |  value: `TextOverlayPosition`         |
|  `TextOverlay`  |  `text`          |  `viewer.text_overlay.events.text`       |                                  |  value: `str`                         |
|  `ViewerModel`  |  `help`          |  `viewer.events.help`                    |                                  |  value: `str`                         |
|  `ViewerModel`  |  `status`        |  `viewer.events.status`                  |                                  |  value: `str`                         |
|  `ViewerModel`  |  `theme`         |  `viewer.events.theme`                   |                                  |  value: `str`                         |
|  `ViewerModel`  |  `title`         |  `viewer.events.title`                   |                                  |  value: `str`                         |
|  `Viewer`       |  `help`          |  `viewer.events.help`                    |                                  |  value: `str`                         |
|  `Viewer`       |  `status`        |  `viewer.events.status`                  |                                  |  value: `str`                         |
|  `Viewer`       |  `theme`         |  `viewer.events.theme`                   |                                  |  value: `str`                         |
|  `Viewer`       |  `title`         |  `viewer.events.title`                   |                                  |  value: `str`                         |
                                                                                                                                                            
<!-- STOP VIEWER EVENTS TABLE -->

## Layer Events

<span style="font-size:0.8em;">
<!-- LAYER EVENTS TABLE -->

<!-- STOP LAYER EVENTS TABLE -->