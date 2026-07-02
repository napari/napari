"""Histogram visual for rendering histogram data in vispy."""

from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh


class HistogramVisual(Compound):
    """
    Visual for rendering histogram data.

    This visual renders a histogram as a bar chart using a mesh,
    with a unified LUT line overlay that combines clim indicators
    and gamma curve into a single path.

    The histogram is rendered in normalized coordinates:
    - X axis: 0 to 1 (bin positions normalized to data range)
    - Y axis: 0 to 1 (counts normalized to max count)

    Actual positioning and sizing in screen space is handled by
    transforms.
    """

    def __init__(self) -> None:
        """Initialize the histogram visual with all sub-components."""
        # Initialize data attributes before creating visuals (vispy freezes
        # objects)
        self._gamma = 1.0
        self._clims: tuple[float, float] | None = None
        self._data_range: tuple[float, float] | None = None
        self._bar_color = (0.8, 0.8, 0.8, 0.8)  # Light gray bars
        self._lut_color = (1.0, 1.0, 0.0, 0.9)  # Yellow for LUT line
        self._axes_color = (0.5, 0.5, 0.5, 1.0)

        # Create sub-visuals
        self._bars = Mesh()
        self._lut_line = Line(method='gl', antialias=True)
        self._axes = Line(method='gl', antialias=True)

        # Set rendering order (higher = on top)
        self._bars.order = 0
        self._axes.order = 1
        self._lut_line.order = 10

        # Disable depth test for translucent rendering
        self._bars.set_gl_state('translucent', depth_test=False)
        self._lut_line.set_gl_state('translucent', depth_test=False)

        # Initialize with empty data to prevent Vispy errors
        self._set_empty_data()

        super().__init__([self._bars, self._axes, self._lut_line])

    def set_data(
        self,
        bins: np.ndarray | None = None,
        counts: np.ndarray | None = None,
        gamma: float = 1.0,
        clims: tuple[float, float] | None = None,
        data_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Update histogram visualization data.

        Parameters
        ----------
        bins : np.ndarray, optional
            Bin edges from histogram computation.
        counts : np.ndarray, optional
            Count values for each bin.
        gamma : float, optional
            Gamma correction value for gamma curve overlay.
        clims : tuple[float, float], optional
            Contrast limits (min, max) for drawing indicator lines.
        data_range : tuple[float, float], optional
            Full data range (min, max) for normalizing positions.
        """
        if (
            bins is None
            or counts is None
            or len(bins) == 0
            or len(counts) == 0
        ):
            self._clear()
            return

        self._gamma = gamma
        self._clims = clims
        self._data_range = data_range

        self._update_bars(bins, counts)
        self._update_axes()

        if clims is not None and data_range is not None:
            self._update_lut_line(clims, gamma, data_range)
        else:
            dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
            self._lut_line.set_data(pos=dummy_line)

    def _set_empty_data(self) -> None:
        """Set minimal valid data to avoid vispy / GL errors."""
        dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
        # Three distinct dummy vertices so vispy never passes a NULL
        # vertex buffer to glVertexAttribPointer (crashes on GL context
        # re-initialization).
        dummy_verts = np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32
        )
        dummy_faces = np.array([[0, 1, 2]], dtype=np.uint32)
        self._bars.set_data(vertices=dummy_verts, faces=dummy_faces)
        self._lut_line.set_data(pos=dummy_line)
        self._axes.set_data(pos=dummy_line)

    def _clear(self) -> None:
        """Clear all visual elements."""
        self._gamma = 1.0
        self._clims = None
        self._data_range = None
        self._set_empty_data()

    def _update_bars(self, bins: np.ndarray, counts: np.ndarray) -> None:
        """Update the bar chart mesh from bins and counts."""
        if len(bins) < 2 or len(counts) == 0:
            self._set_empty_data()
            return

        # Normalize to [0, 1]
        bin_min = bins[0]
        bin_range = bins[-1] - bin_min
        if bin_range == 0:
            bin_range = 1

        max_count = float(np.max(counts))
        if max_count == 0:
            max_count = 1

        # Create vertices and faces for the bar mesh
        # Each bar is represented as two triangles (4 vertices, 2 faces).
        n_bins = len(counts)
        x_lefts = (bins[:n_bins] - bin_min) / bin_range
        x_rights = (bins[1:] - bin_min) / bin_range
        y_tops = counts.astype(np.float32) / max_count

        # Build mesh
        vertices = np.empty((n_bins * 4, 3), dtype=np.float32)
        faces = np.empty((n_bins * 2, 3), dtype=np.uint32)
        for i in range(n_bins):
            v0 = i * 4
            vertices[v0] = [x_lefts[i], 0.0, 0.0]
            vertices[v0 + 1] = [x_rights[i], 0.0, 0.0]
            vertices[v0 + 2] = [x_rights[i], y_tops[i], 0.0]
            vertices[v0 + 3] = [x_lefts[i], y_tops[i], 0.0]
            f0 = i * 2
            faces[f0] = [v0, v0 + 1, v0 + 2]
            faces[f0 + 1] = [v0, v0 + 2, v0 + 3]

        face_colors = np.tile(self._bar_color, (len(faces), 1)).astype(
            np.float32
        )
        self._bars.set_data(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
        )

    def _update_axes(self) -> None:
        """Draw simple X and Y axes at bottom and left edges."""
        axes_pos = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        connect = np.array([[0, 1], [0, 2]], dtype=np.uint32)
        self._axes.set_data(
            pos=axes_pos,
            connect=connect,
            color=self._axes_color,
            width=1.0,
        )

    def _update_lut_line(
        self,
        clims: tuple[float, float],
        gamma: float,
        data_range: tuple[float, float],
        npoints: int = 256,
    ) -> None:
        """
        Draw unified LUT line combining clim indicators and gamma curve.

        This creates a single connected line path:
        1. Bottom-left clim (vertical line from 0 to 1)
        2. Top-left clim
        3. Gamma curve (npoints from clim_min to clim_max)
        4. Top-right clim
        5. Bottom-right clim (vertical line from 1 to 0)
        """
        clim_min, clim_max = clims
        data_min, data_max = data_range
        range_size = data_max - data_min
        if range_size == 0:
            range_size = 1

        x_min = np.clip((clim_min - data_min) / range_size, 0, 1)
        x_max = np.clip((clim_max - data_min) / range_size, 0, 1)

        x_coords: list[float] = [x_min, x_min]
        y_coords: list[float] = [0.0, 1.0]

        if x_max > x_min:
            x_gamma = np.linspace(x_min, x_max, npoints)
            y_gamma = np.power((x_gamma - x_min) / (x_max - x_min), gamma)
            x_coords.extend(x_gamma.tolist())
            y_coords.extend(y_gamma.tolist())
        else:
            x_coords.append(x_min)
            y_coords.append(0.5)

        x_coords.extend([x_max, x_max])
        y_coords.extend([1.0, 0.0])

        # Combine into 2D positions (vispy Line with method='gl' expects 2D)
        pos = np.column_stack([x_coords, y_coords]).astype(np.float32)
        self._lut_line.set_data(pos=pos, color=self._lut_color, width=2.0)

    def set_style(
        self,
        bar_color: tuple[float, float, float, float] | None = None,
        lut_color: tuple[float, float, float, float] | None = None,
        axes_color: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Set colours for visual elements."""
        if bar_color is not None:
            self._bar_color = bar_color
        if lut_color is not None:
            self._lut_color = lut_color
        if axes_color is not None:
            self._axes_color = axes_color

    def destroy(self) -> None:
        """Clean up visual resources to avoid vispy resource leaks."""
        for child in self._subvisuals:
            if hasattr(child, 'parent') and child.parent is not None:
                child.parent = None
