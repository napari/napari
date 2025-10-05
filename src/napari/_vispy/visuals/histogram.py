"""Histogram visual for rendering histogram data in vispy."""

from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text


class HistogramVisual(Compound):
    """
    Visual for rendering histogram data.

    This visual renders a histogram as a bar chart using a mesh,
    with optional overlay elements like gamma curves and contrast
    limit indicators.

    The histogram is rendered in normalized coordinates:
    - X axis: 0 to 1 (bin positions normalized to data range)
    - Y axis: 0 to 1 (counts normalized to max count)

    Actual positioning and sizing in screen space is handled by
    the VispyHistogramOverlay class via transforms.
    """

    def __init__(self) -> None:
        """Initialize the histogram visual with all sub-components."""
        # Initialize data attributes before creating visuals (vispy freezes objects)
        self._bins = np.array([])
        self._counts = np.array([])
        self._log_scale = False
        self._bar_color = (0.8, 0.8, 0.8, 0.8)
        self._gamma_color = (1.0, 1.0, 0.0, 0.9)  # Yellow for gamma curve
        self._clim_color = (1.0, 1.0, 0.0, 0.9)  # Yellow for contrast limits
        self._axes_color = (0.5, 0.5, 0.5, 1.0)

        # Create sub-visuals
        self._bars = Mesh()
        self._gamma_line = Line(method='gl', antialias=True)
        self._clim_lines = Line(method='gl', antialias=True)
        self._axes = Line(method='gl', antialias=True)
        self._text = Text(color='white', font_size=8, anchor_x='left')

        # Initialize lines with dummy 2D data (vispy requires proper shape)
        # Use a simple line that won't be visible
        dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
        self._gamma_line.set_data(pos=dummy_line)
        self._clim_lines.set_data(pos=dummy_line)
        self._axes.set_data(pos=dummy_line)

        # Combine into compound visual - order matters for rendering!
        # Draw bars and axes first, then gamma/clims on top
        super().__init__(
            [
                self._axes,
                self._bars,
                self._gamma_line,
                self._clim_lines,
                self._text,
            ]
        )

    @property
    def bars(self) -> Mesh:
        """Get the mesh visual for histogram bars."""
        return self._bars

    @property
    def gamma_line(self) -> Line:
        """Get the line visual for gamma curve."""
        return self._gamma_line

    @property
    def clim_lines(self) -> Line:
        """Get the line visual for contrast limit indicators."""
        return self._clim_lines

    @property
    def axes(self) -> Line:
        """Get the line visual for axes."""
        return self._axes

    @property
    def text(self) -> Text:
        """Get the text visual for labels."""
        return self._text

    def set_data(
        self,
        bins: np.ndarray | None = None,
        counts: np.ndarray | None = None,
        log_scale: bool = False,
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
        log_scale : bool, optional
            Whether counts are in log scale.
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
            # Clear visualization if no data
            self._clear()
            return

        self._bins = bins
        self._counts = counts
        self._log_scale = log_scale

        # Update bar chart
        self._update_bars()

        # Update axes
        self._update_axes()

        # Always update gamma curve (even when gamma=1.0, shows straight line)
        self._update_gamma_curve(gamma)

        # Update contrast limit indicators if provided
        if clims is not None and data_range is not None:
            self._update_clim_lines(clims, data_range)
        else:
            # Use dummy line instead of empty array
            self._clim_lines.set_data(
                pos=np.array([[0, 0], [0, 0]], dtype=np.float32)
            )

    def _clear(self) -> None:
        """Clear all visual elements."""
        dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
        self._bars.set_data(vertices=np.array([]), faces=np.array([]))
        self._gamma_line.set_data(pos=dummy_line)
        self._clim_lines.set_data(pos=dummy_line)
        self._axes.set_data(pos=dummy_line)

    def _update_bars(self) -> None:
        """Update the bar chart mesh from bins and counts."""
        if len(self._bins) < 2 or len(self._counts) == 0:
            self._bars.set_data(vertices=np.array([]), faces=np.array([]))
            return

        # Normalize data to [0, 1] range
        # X: bin positions normalized by data range
        bin_min = self._bins[0]
        bin_max = self._bins[-1]
        bin_range = bin_max - bin_min
        if bin_range == 0:
            bin_range = 1

        # Y: counts normalized by max count
        max_count = np.max(self._counts)
        if max_count == 0:
            max_count = 1

        # Create vertices and faces for bar chart
        # Each bar is a rectangle (2 triangles = 6 vertices)
        n_bins = len(self._counts)
        vertices = []
        faces = []

        for i in range(n_bins):
            # Bin edges in normalized coordinates
            x_left = (self._bins[i] - bin_min) / bin_range
            x_right = (self._bins[i + 1] - bin_min) / bin_range
            y_bottom = 0.0
            y_top = self._counts[i] / max_count

            # Create 4 vertices for this bar (rectangle)
            v_idx = i * 4
            vertices.extend(
                [
                    [x_left, y_bottom, 0],
                    [x_right, y_bottom, 0],
                    [x_right, y_top, 0],
                    [x_left, y_top, 0],
                ]
            )

            # Create 2 triangles for this bar
            faces.extend(
                [
                    [v_idx, v_idx + 1, v_idx + 2],
                    [v_idx, v_idx + 2, v_idx + 3],
                ]
            )

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)

        # Set mesh data with uniform color
        face_colors = np.tile(self._bar_color, (len(faces), 1))
        self._bars.set_data(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
        )

    def _update_axes(self) -> None:
        """Draw simple X and Y axes at bottom and left edges."""
        # Simple axes: bottom edge (X) and left edge (Y)
        axes_pos = np.array(
            [
                [0, 0, 0],  # Bottom-left corner
                [1, 0, 0],  # Bottom-right corner
                [0, 0, 0],  # Bottom-left corner (repeated for Y axis)
                [0, 1, 0],  # Top-left corner
            ],
            dtype=np.float32,
        )

        connect = np.array([[0, 1], [2, 3]], dtype=np.uint32)

        self._axes.set_data(
            pos=axes_pos,
            connect=connect,
            color=self._axes_color,
            width=1.0,
        )

    def _update_gamma_curve(self, gamma: float) -> None:
        """
        Draw gamma correction curve overlay.

        Parameters
        ----------
        gamma : float
            Gamma correction value.
        """
        # Generate gamma curve: y = x^gamma
        x = np.linspace(0, 1, 100)
        y = np.power(x, gamma)

        # Create line positions
        pos = np.column_stack([x, y, np.zeros_like(x)]).astype(np.float32)

        self._gamma_line.set_data(
            pos=pos,
            color=self._gamma_color,
            width=2.0,
        )

    def _update_clim_lines(
        self, clims: tuple[float, float], data_range: tuple[float, float]
    ) -> None:
        """
        Draw vertical lines indicating contrast limits.

        Parameters
        ----------
        clims : tuple[float, float]
            Contrast limits (min, max).
        data_range : tuple[float, float]
            Full data range (min, max) for normalization.
        """
        clim_min, clim_max = clims
        data_min, data_max = data_range
        range_size = data_max - data_min
        if range_size == 0:
            range_size = 1

        # Normalize contrast limits to [0, 1]
        x_min = (clim_min - data_min) / range_size
        x_max = (clim_max - data_min) / range_size

        # Clamp to [0, 1] range
        x_min = np.clip(x_min, 0, 1)
        x_max = np.clip(x_max, 0, 1)

        # Create vertical lines at clim positions
        pos = np.array(
            [
                [x_min, 0, 0],
                [x_min, 1, 0],
                [x_max, 0, 0],
                [x_max, 1, 0],
            ],
            dtype=np.float32,
        )

        connect = np.array([[0, 1], [2, 3]], dtype=np.uint32)

        self._clim_lines.set_data(
            pos=pos,
            connect=connect,
            color=self._clim_color,
            width=2.0,
        )

    def set_colors(
        self,
        bar_color: tuple[float, float, float, float] | None = None,
        gamma_color: tuple[float, float, float, float] | None = None,
        clim_color: tuple[float, float, float, float] | None = None,
        axes_color: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Set colors for different visual elements.

        Parameters
        ----------
        bar_color : tuple, optional
            RGBA color for histogram bars.
        gamma_color : tuple, optional
            RGBA color for gamma curve.
        clim_color : tuple, optional
            RGBA color for contrast limit lines.
        axes_color : tuple, optional
            RGBA color for axes.
        """
        if bar_color is not None:
            self._bar_color = bar_color
        if gamma_color is not None:
            self._gamma_color = gamma_color
        if clim_color is not None:
            self._clim_color = clim_color
        if axes_color is not None:
            self._axes_color = axes_color

        # Reapply colors if data exists
        if len(self._counts) > 0:
            self._update_bars()
            self._update_axes()
