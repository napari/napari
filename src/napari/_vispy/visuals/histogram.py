"""Histogram visual for rendering histogram data in vispy."""

from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text


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
    the VispyHistogramOverlay class via transforms.
    """

    def __init__(self) -> None:
        """Initialize the histogram visual with all sub-components."""
        # Initialize data attributes before creating visuals (vispy freezes objects)
        self._bins = np.array([])
        self._counts = np.array([])
        self._log_scale = False
        self._bar_color = (0.8, 0.8, 0.8, 0.8)
        self._lut_color = (1.0, 1.0, 0.0, 0.9)  # Yellow for unified LUT line
        self._axes_color = (0.5, 0.5, 0.5, 1.0)

        # Create sub-visuals
        self._bars = Mesh()
        self._lut_line = Line(method='gl', antialias=True)  # Unified LUT line
        self._axes = Line(method='gl', antialias=True)
        self._text = Text(color='white', font_size=8, anchor_x='left')

        # Set rendering order (higher = on top)
        self._bars.order = 0  # Draw first (behind everything)
        self._axes.order = 1  # Draw above bars
        self._lut_line.order = 10  # Draw on top of histogram
        self._text.order = 20  # Draw on top of everything

        # Disable depth test for LUT line to ensure it renders on top
        self._lut_line.set_gl_state('translucent', depth_test=False)

        # Initialize lines with dummy 2D data (vispy requires proper shape)
        # Use a simple line that won't be visible
        dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
        self._lut_line.set_data(pos=dummy_line)
        self._axes.set_data(pos=dummy_line)

        # Combine into compound visual - order matters for rendering!
        # Draw bars and axes first, then LUT line on top
        super().__init__(
            [
                self._bars,
                self._axes,
                self._lut_line,
                self._text,
            ]
        )

    @property
    def bars(self) -> Mesh:
        """Get the mesh visual for histogram bars."""
        return self._bars

    @property
    def lut_line(self) -> Line:
        """Get the unified LUT line visual (clims + gamma curve)."""
        return self._lut_line

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

        # Update unified LUT line (combines clims and gamma curve)
        if clims is not None and data_range is not None:
            self._update_lut_line(clims, gamma, data_range)
        else:
            # Use dummy line instead of empty array
            dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
            self._lut_line.set_data(pos=dummy_line)

    def _clear(self) -> None:
        """Clear all visual elements."""
        dummy_line = np.array([[0, 0], [0, 0]], dtype=np.float32)
        # Use minimal valid mesh instead of empty arrays to avoid vispy errors
        dummy_vertices = np.array([[0, 0, 0]], dtype=np.float32)
        dummy_faces = np.array([[0, 0, 0]], dtype=np.uint32)
        self._bars.set_data(vertices=dummy_vertices, faces=dummy_faces)
        self._lut_line.set_data(pos=dummy_line)
        self._axes.set_data(pos=dummy_line)

    def _update_bars(self) -> None:
        """Update the bar chart mesh from bins and counts."""
        if len(self._bins) < 2 or len(self._counts) == 0:
            # Use minimal valid mesh instead of empty arrays to avoid vispy errors
            dummy_vertices = np.array([[0, 0, 0]], dtype=np.float32)
            dummy_faces = np.array([[0, 0, 0]], dtype=np.uint32)
            self._bars.set_data(vertices=dummy_vertices, faces=dummy_faces)
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

    def _update_lut_line(
        self,
        clims: tuple[float, float],
        gamma: float,
        data_range: tuple[float, float],
        npoints: int = 256,
    ) -> None:
        """
        Draw unified LUT line combining clim indicators and gamma curve.

        This creates a single connected line path following the pattern:
        1. Bottom-left clim (vertical line from 0 to 1)
        2. Top-left clim
        3. Gamma curve (npoints from clim_min to clim_max)
        4. Top-right clim
        5. Bottom-right clim (vertical line from 1 to 0)

        Total vertices: npoints + 4

        Parameters
        ----------
        clims : tuple[float, float]
            Contrast limits (min, max) in data coordinates.
        gamma : float
            Gamma correction value for the curve.
        data_range : tuple[float, float]
            Full data range (min, max) for normalization.
        npoints : int, default: 256
            Number of points for the gamma curve.
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

        # Build the unified LUT line path:
        # Start with left clim vertical line (bottom to top)
        x_coords = [x_min, x_min]
        y_coords = [0.0, 1.0]

        # Add gamma curve points
        if x_max > x_min:
            x_gamma = np.linspace(x_min, x_max, npoints)
            # Normalize to 0-1 for gamma calculation
            x_norm = (x_gamma - x_min) / (x_max - x_min)
            # Apply gamma: y = x^gamma
            y_gamma = np.power(x_norm, gamma)
            x_coords.extend(x_gamma.tolist())
            y_coords.extend(y_gamma.tolist())
        else:
            # If clims are equal or inverted, just draw vertical line
            x_coords.append(x_min)
            y_coords.append(0.5)

        # End with right clim vertical line (top to bottom)
        x_coords.extend([x_max, x_max])
        y_coords.extend([1.0, 0.0])

        # Combine into 3D positions (Z=0 for all points)
        pos = np.column_stack(
            [x_coords, y_coords, np.zeros(len(x_coords))]
        ).astype(np.float32)

        # Set line data with strip connection (all points connected in sequence)
        # Use solid color - no gradient/pattern
        self._lut_line.set_data(
            pos=pos,
            color=self._lut_color,
            width=2.0,
        )

    def set_colors(
        self,
        bar_color: tuple[float, float, float, float] | None = None,
        lut_color: tuple[float, float, float, float] | None = None,
        axes_color: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Set colors for different visual elements.

        Parameters
        ----------
        bar_color : tuple, optional
            RGBA color for histogram bars.
        lut_color : tuple, optional
            RGBA color for unified LUT line (clims + gamma curve).
        axes_color : tuple, optional
            RGBA color for axes.
        """
        if bar_color is not None:
            self._bar_color = bar_color
        if lut_color is not None:
            self._lut_color = lut_color
        if axes_color is not None:
            self._axes_color = axes_color

        # Reapply colors if data exists
        if len(self._counts) > 0:
            self._update_bars()
            self._update_axes()
