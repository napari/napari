# Events Reference


## Viewer Events

<span style="font-size:0.8em;">
<!-- VIEWER EVENTS TABLE -->
                                                                                                                                                            
|  Class          |  Event Name      |  Access At                               |  Emitted when __ changes         |  Event Attribute(s)                   |
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
</span>

## Layer Events

Access all `Layer` events, at `layer.events.<event_name>`

<span style="font-size:0.8em;">
<!-- LAYER EVENTS TABLE -->
                                                                                           
|  Class      |  Event Name            |  Emitted when __ changes  |  Event Attribute(s)  |
|-------------|------------------------|---------------------------|----------------------|
|  `Layer`    |  `source`              |                           |                      |
|  `Layer`    |  `auto_connect`        |                           |                      |
|  `Layer`    |  `refresh`             |                           |                      |
|  `Layer`    |  `set_data`            |                           |                      |
|  `Layer`    |  `blending`            |                           |                      |
|  `Layer`    |  `opacity`             |                           |                      |
|  `Layer`    |  `visible`             |                           |                      |
|  `Layer`    |  `scale`               |                           |                      |
|  `Layer`    |  `translate`           |                           |                      |
|  `Layer`    |  `rotate`              |                           |                      |
|  `Layer`    |  `shear`               |                           |                      |
|  `Layer`    |  `affine`              |                           |                      |
|  `Layer`    |  `data`                |                           |                      |
|  `Layer`    |  `name`                |                           |                      |
|  `Layer`    |  `thumbnail`           |                           |                      |
|  `Layer`    |  `status`              |                           |                      |
|  `Layer`    |  `help`                |                           |                      |
|  `Layer`    |  `interactive`         |                           |                      |
|  `Layer`    |  `cursor`              |                           |                      |
|  `Layer`    |  `cursor_size`         |                           |                      |
|  `Layer`    |  `editable`            |                           |                      |
|  `Layer`    |  `loaded`              |                           |                      |
|  `Layer`    |  `_ndisplay`           |                           |                      |
|  `Layer`    |  `select`              |                           |                      |
|  `Layer`    |  `deselect`            |                           |                      |
|  `Image`    |  `contrast_limits`     |                           |                      |
|  `Image`    |  `gamma`               |                           |                      |
|  `Image`    |  `colormap`            |                           |                      |
|  `Image`    |  `interpolation`       |                           |                      |
|  `Image`    |  `rendering`           |                           |                      |
|  `Image`    |  `iso_threshold`       |                           |                      |
|  `Image`    |  `attenuation`         |                           |                      |
|  `Labels`   |  `contrast_limits`     |                           |                      |
|  `Labels`   |  `gamma`               |                           |                      |
|  `Labels`   |  `colormap`            |                           |                      |
|  `Labels`   |  `interpolation`       |                           |                      |
|  `Labels`   |  `rendering`           |                           |                      |
|  `Labels`   |  `iso_threshold`       |                           |                      |
|  `Labels`   |  `attenuation`         |                           |                      |
|  `Labels`   |  `mode`                |                           |                      |
|  `Labels`   |  `preserve_labels`     |                           |                      |
|  `Labels`   |  `properties`          |                           |                      |
|  `Labels`   |  `n_dimensional`       |                           |                      |
|  `Labels`   |  `n_edit_dimensions`   |                           |                      |
|  `Labels`   |  `contiguous`          |                           |                      |
|  `Labels`   |  `brush_size`          |                           |                      |
|  `Labels`   |  `selected_label`      |                           |                      |
|  `Labels`   |  `color_mode`          |                           |                      |
|  `Labels`   |  `brush_shape`         |                           |                      |
|  `Labels`   |  `contour`             |                           |                      |
|  `Points`   |  `mode`                |                           |                      |
|  `Points`   |  `size`                |                           |                      |
|  `Points`   |  `edge_width`          |                           |                      |
|  `Points`   |  `face_color`          |                           |                      |
|  `Points`   |  `current_face_color`  |                           |                      |
|  `Points`   |  `edge_color`          |                           |                      |
|  `Points`   |  `current_edge_color`  |                           |                      |
|  `Points`   |  `properties`          |                           |                      |
|  `Points`   |  `current_properties`  |                           |                      |
|  `Points`   |  `symbol`              |                           |                      |
|  `Points`   |  `n_dimensional`       |                           |                      |
|  `Points`   |  `highlight`           |                           |                      |
|  `Vectors`  |  `length`              |                           |                      |
|  `Vectors`  |  `edge_width`          |                           |                      |
|  `Vectors`  |  `edge_color`          |                           |                      |
|  `Vectors`  |  `edge_color_mode`     |                           |                      |
|  `Vectors`  |  `properties`          |                           |                      |
|  `Shapes`   |  `mode`                |                           |                      |
|  `Shapes`   |  `edge_width`          |                           |                      |
|  `Shapes`   |  `edge_color`          |                           |                      |
|  `Shapes`   |  `face_color`          |                           |                      |
|  `Shapes`   |  `properties`          |                           |                      |
|  `Shapes`   |  `current_edge_color`  |                           |                      |
|  `Shapes`   |  `current_face_color`  |                           |                      |
|  `Shapes`   |  `current_properties`  |                           |                      |
|  `Shapes`   |  `highlight`           |                           |                      |
|  `Surface`  |  `contrast_limits`     |                           |                      |
|  `Surface`  |  `gamma`               |                           |                      |
|  `Surface`  |  `colormap`            |                           |                      |
|  `Surface`  |  `interpolation`       |                           |                      |
|  `Surface`  |  `rendering`           |                           |                      |
|  `Tracks`   |  `tail_width`          |                           |                      |
|  `Tracks`   |  `tail_length`         |                           |                      |
|  `Tracks`   |  `display_id`          |                           |                      |
|  `Tracks`   |  `display_tail`        |                           |                      |
|  `Tracks`   |  `display_graph`       |                           |                      |
|  `Tracks`   |  `color_by`            |                           |                      |
|  `Tracks`   |  `colormap`            |                           |                      |
|  `Tracks`   |  `properties`          |                           |                      |
|  `Tracks`   |  `rebuild_tracks`      |                           |                      |
|  `Tracks`   |  `rebuild_graph`       |                           |                      |
                                                                                           
<!-- STOP LAYER EVENTS TABLE -->
</span>