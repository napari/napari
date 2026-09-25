"""
Projection modes for points
===========================

A ``Points`` layer can show the points that are not in the current slice, and
``projection_mode`` determines how those points are displayed.
``Points`` are drawn as discs, but are only really coordinates, and some folks
use points to represent ndimensional visuals (e.g. a sphere with the diamater
of the point's size) and are thus interested in the extent of the point rather
than only its position.

This example draws one layer per available points projection mode, over a
grid of points that is offset in the two axes that are not displayed: time and z.

The rows come in groups, one group per mode, and within a group one row per time
offset; each column is a z offset. Every point therefore sits in its own place on
screen, and its position tells you where it is in both axes. Both offsets are in
units of the point radius, ``r = size / 2``:

    z = -1.5r  ...  z = 0  ...  z = 1.5r    columns, left to right
    mode 1  t = 0                           rows, top to bottom, in groups of
    mode 1  t = 0.5r                        three, one group per mode
    mode 1  t = 1.5r
    mode 2  t = 0
    mode 2  t = 0.5r
    .... etc

Drag the first dims slider to move the slice plane in z and the second to move
it in time. Right-click a dims slider handle to increase the slice thickness.

With no slice thickness, ``none`` and ``all`` show only the single point that is
exactly on the plane, while the two ``_ND`` modes show every point whose own
extent reaches the plane, which is up to one radius away in both axes. That is
why the ``t = 1.5r`` rows stay empty until the slice is thickened in t.

The modes:

* ``none``: only the points on the plane, so a row empties as soon as the plane
  moves past it, regardless of slice thickness.
* ``all``: every point inside the thick slice, at its full size.
* ``rescale_linear``: like ``all``, but points shrink towards the face of the
  slice, so they need a thick slice to show up at all.
* ``rescale_spherical``: like ``rescale_linear``, but the size is the
  cross-section of the point's spherical extent rather than a linear fade.
* ``rescale_linear_nd`` and ``rescale_spherical_nd``: the point is treated as an
  object with an extent the size of its diameter instead of as a position, so it
  is shown when that extent reaches the slice, and it keeps its full size inside
  the slice. ``rescale_linear_nd`` directly replace the deprecated <0.9
  ``out_of_slice_display``.

Point text is always attached to a single point, so the row labels are carried
by the point at ``z = 0`` of each row.
"""

import numpy as np

import napari

SIZE = 20.0
RADIUS = SIZE / 2
COL_SPACING = 3 * SIZE  # z offset, left to right
ROW_SPACING = 2.6 * SIZE  # mode and time offset, top to bottom
MODES = {
    'none': 'white',
    'all': 'cyan',
    'rescale_linear': 'magenta',
    'rescale_linear_nd': 'lime',
    'rescale_spherical': 'orange',
    'rescale_spherical_nd': 'deepskyblue',
}

# offset from the slice plane, in point radii: one per column (z) and one per
# row of each group (t)
Z_OFFSETS = np.linspace(-1.5, 1.5, 13)
TIME_OFFSETS = np.array([0.0, 0.5, 1.5])
N_COLS = len(Z_OFFSETS)
MIDDLE_COL = N_COLS // 2  # z = 0

viewer = napari.Viewer()

for mode_index, (mode, color) in enumerate(MODES.items()):
    rows = []
    labels = []
    for block, t_offset in enumerate(TIME_OFFSETS):
        y = (mode_index * len(TIME_OFFSETS) + block) * ROW_SPACING
        rows.append(
            np.column_stack(
                [
                    np.full(N_COLS, t_offset * RADIUS),
                    Z_OFFSETS * RADIUS,
                    np.full(N_COLS, y),
                    np.arange(N_COLS) * COL_SPACING,
                ]
            )
        )
        row_labels = [''] * N_COLS
        row_labels[MIDDLE_COL] = f'{mode} t={t_offset:+g}r'
        labels.extend(row_labels)
    viewer.add_points(
        np.concatenate(rows),
        name=mode,
        size=SIZE,
        face_color=color,
        projection_mode=mode,
        # the viewer takes its dims axis labels from the layers
        axis_labels=('t', 'z', 'y', 'x'),
        text={
            'string': labels,
            'anchor': 'upper_left',
            'color': 'white',
            'size': 9,
        },
    )

# the slice plane sits at t = 0 and z = 0, the middle column of the first row
viewer.dims.set_point(0, 0)
viewer.dims.set_point(1, 0)
viewer.fit_to_view()

if __name__ == '__main__':
    napari.run()
