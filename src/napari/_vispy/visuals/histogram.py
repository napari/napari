"""Simple VisPy-based histogram visual for contrast limit visualization.

Inspired by ndv's histogram implementation, but simplified for napari's use case.
Focus is on performance over visual quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene.visuals import Line, Markers, Mesh

if TYPE_CHECKING:
    from collections.abc import Sequence


def _hist_counts_to_mesh(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    orientation: str = 'vertical',
) -> tuple[np.ndarray, np.ndarray]:
    """Convert histogram counts and bin edges to mesh vertices and faces.

    This is adapted from vispy's histogram visual.

    Parameters
    ----------
    counts : np.ndarray
        Histogram bin counts
    bin_edges : np.ndarray
        Histogram bin edges (length = counts + 1)
    orientation : str
        'vertical' for bars going up, 'horizontal' for bars going right

    Returns
    -------
    vertices : np.ndarray
        Mesh vertices (Nx3)
    faces : np.ndarray
        Mesh faces (Mx3)
    """
    # Create vertices for rectangles
    n_bins = len(counts)
    vertices = np.zeros((n_bins * 4, 3), dtype=np.float32)
    faces = np.zeros((n_bins * 2, 3), dtype=np.uint32)

    for i in range(n_bins):
        x0, x1 = bin_edges[i], bin_edges[i + 1]
        y0, y1 = 0, counts[i]

        if orientation == 'vertical':
            # Bottom-left, bottom-right, top-right, top-left
            vertices[i * 4 + 0] = [x0, y0, 0]
            vertices[i * 4 + 1] = [x1, y0, 0]
            vertices[i * 4 + 2] = [x1, y1, 0]
            vertices[i * 4 + 3] = [x0, y1, 0]
        else:  # horizontal
            vertices[i * 4 + 0] = [y0, x0, 0]
            vertices[i * 4 + 1] = [y0, x1, 0]
            vertices[i * 4 + 2] = [y1, x1, 0]
            vertices[i * 4 + 3] = [y1, x0, 0]

        # Two triangles per rectangle
        base_idx = i * 4
        faces[i * 2 + 0] = [base_idx, base_idx + 1, base_idx + 2]
        faces[i * 2 + 1] = [base_idx, base_idx + 2, base_idx + 3]

    return vertices, faces


class HistogramVisual:
    """Simple histogram visual using VisPy for performance.

    This creates a mesh-based histogram and optional vertical lines
    for contrast limit indicators.

    Parameters
    ----------
    parent : vispy.scene.Node, optional
        Parent node for the histogram visual
    color : str or tuple
        Color for histogram bars
    orientation : str
        'vertical' or 'horizontal'

    Attributes
    ----------
    mesh : vispy.scene.visuals.Mesh
        The histogram mesh visual
    clim_lines : vispy.scene.visuals.Line
        Lines indicating contrast limits
    """

    def __init__(
        self,
        parent=None,
        color: str | tuple = '#888888',
        orientation: str = 'vertical',
        log_scale: bool = False,
    ) -> None:
        self.orientation = orientation
        self.log_scale = log_scale
        self._bin_edges: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._gamma: float = 1.0
        self._clims: tuple[float, float] | None = None

        # Create mesh for histogram bars
        self.mesh = Mesh(color=color)
        if parent is not None:
            self.mesh.parent = parent
        self.mesh.order = 0  # Draw first (behind)

        # Create gamma curve line connecting the contrast limits
        self.gamma_line = Line(color='cyan', width=2, connect='strip')
        if parent is not None:
            self.gamma_line.parent = parent
        self.gamma_line.order = 9  # Draw above histogram, below clim lines
        self.gamma_line.set_gl_state('translucent', depth_test=False)
        self.gamma_line.visible = False

        # Create gamma handle marker (draggable indicator)
        self.gamma_handle = Markers()
        self.gamma_handle.set_data(
            pos=np.array([[0, 0, 0.06]]),  # z slightly above gamma line
            size=8,
            edge_width=1.5,
            edge_color='cyan',
            face_color='black',
        )
        if parent is not None:
            self.gamma_handle.parent = parent
        self.gamma_handle.order = 11  # Draw on top of everything
        self.gamma_handle.set_gl_state('translucent', depth_test=False)
        self.gamma_handle.visible = False

        # Create lines for contrast limit indicators
        self.clim_lines = Line(color='yellow', width=2, connect='segments')
        if parent is not None:
            self.clim_lines.parent = parent
        self.clim_lines.order = 10  # Draw on top (higher value = on top)
        # Disable depth test so lines always render on top
        self.clim_lines.set_gl_state('translucent', depth_test=False)

    def set_data(
        self,
        data: np.ndarray | None = None,
        bins: int | Sequence | np.ndarray = 128,
        data_range: tuple[float, float] | None = None,
    ) -> None:
        """Compute and display histogram from data.

        Parameters
        ----------
        data : np.ndarray, optional
            Data to compute histogram from. If None, clears histogram.
        bins : int or sequence
            Number of bins or bin edges
        data_range : tuple of float, optional
            (min, max) range for histogram. If None, uses data min/max
        """
        if data is None or len(data) == 0:
            self.mesh.visible = False
            return

        # Compute histogram
        # For performance, sample large arrays
        if data.size > 1_000_000:
            # Sample uniformly
            step = max(1, data.size // 1_000_000)
            data = data.ravel()[::step]
        else:
            data = data.ravel()

        # Remove NaN and Inf
        data = data[np.isfinite(data)]

        if len(data) == 0:
            self.mesh.visible = False
            return

        self._counts, self._bin_edges = np.histogram(
            data, bins=bins, range=data_range
        )

        # Apply log scaling if enabled
        counts_to_display = self._counts.copy()
        if self.log_scale:
            # Use log(counts + 1) to handle zero counts gracefully
            counts_to_display = np.log10(counts_to_display + 1)

        # Convert to mesh
        vertices, faces = _hist_counts_to_mesh(
            counts_to_display, self._bin_edges, self.orientation
        )

        self.mesh.set_data(vertices=vertices, faces=faces)
        self.mesh.visible = True

    def set_clims(
        self,
        clims: tuple[float, float],
        gamma: float = 1.0,
        log_scale: bool = False,
    ) -> None:
        """Set contrast limit indicator lines and gamma curve.

        Parameters
        ----------
        clims : tuple of float
            (min, max) contrast limits to display
        gamma : float, optional
            Gamma value for the curve connecting clims (default: 1.0)
        log_scale : bool, optional
            Whether histogram is in log scale mode (affects line height)
        """
        if self._counts is None or self._bin_edges is None:
            return

        self._clims = clims
        self._gamma = gamma

        # Calculate max count for display (accounting for log scale)
        max_count = self._counts.max() if len(self._counts) > 0 else 1
        if log_scale:
            max_count = np.log10(max_count + 1)

        if self.orientation == 'vertical':
            # Vertical lines at clim positions with z-offset to render in front
            pos = np.array(
                [
                    [clims[0], 0, 0.1],
                    [clims[0], max_count, 0.1],
                    [clims[1], 0, 0.1],
                    [clims[1], max_count, 0.1],
                ],
                dtype=np.float32,
            )
        else:  # horizontal
            pos = np.array(
                [
                    [0, clims[0], 0.1],
                    [max_count, clims[0], 0.1],
                    [0, clims[1], 0.1],
                    [max_count, clims[1], 0.1],
                ],
                dtype=np.float32,
            )

        self.clim_lines.set_data(pos=pos)
        self.clim_lines.visible = True

        # Update gamma curve
        self._update_gamma_curve()

    def _update_gamma_curve(self) -> None:
        """Update the gamma curve line connecting the contrast limits."""
        if self._clims is None or self._counts is None:
            return

        clim_min, clim_max = self._clims

        # Calculate max count for display (accounting for log scale)
        max_count = self._counts.max() if len(self._counts) > 0 else 1
        if self.log_scale:
            max_count = np.log10(max_count + 1)

        # Generate points along the gamma curve
        n_points = 50
        x_values = np.linspace(clim_min, clim_max, n_points)

        # Normalize to 0-1 range
        if clim_max > clim_min:
            normalized = (x_values - clim_min) / (clim_max - clim_min)
            # Apply gamma curve
            y_normalized = normalized**self._gamma
            # Scale to histogram height
            y_values = y_normalized * max_count
        else:
            y_values = np.zeros(n_points)

        if self.orientation == 'vertical':
            # Points from left clim to right clim following gamma curve
            pos = np.column_stack(
                [x_values, y_values, np.full(n_points, 0.05)]
            )
            # Position handle at midpoint of gamma curve
            mid_idx = n_points // 2
            handle_pos = np.array(
                [[x_values[mid_idx], y_values[mid_idx], 0.06]]
            )
        else:  # horizontal
            pos = np.column_stack(
                [y_values, x_values, np.full(n_points, 0.05)]
            )
            # Position handle at midpoint of gamma curve
            mid_idx = n_points // 2
            handle_pos = np.array(
                [[y_values[mid_idx], x_values[mid_idx], 0.06]]
            )

        self.gamma_line.set_data(pos=pos.astype(np.float32))
        self.gamma_line.visible = True

        # Update gamma handle position
        self.gamma_handle.set_data(pos=handle_pos.astype(np.float32))
        self.gamma_handle.visible = True

    @property
    def visible(self) -> bool:
        """Return visibility of the histogram."""
        return self.mesh.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility of the histogram."""
        self.mesh.visible = value
        self.clim_lines.visible = value
        self.gamma_line.visible = value
        self.gamma_handle.visible = value

    def set_log_scale(self, enabled: bool) -> None:
        """Enable or disable logarithmic scaling.

        Parameters
        ----------
        enabled : bool
            If True, use log10(counts + 1) for display.
            If False, use linear counts.
        """
        if self.log_scale != enabled:
            self.log_scale = enabled
            # Re-compute the mesh with new scaling
            if self._counts is not None and self._bin_edges is not None:
                counts_to_display = self._counts.copy()
                if self.log_scale:
                    counts_to_display = np.log10(counts_to_display + 1)

                vertices, faces = _hist_counts_to_mesh(
                    counts_to_display, self._bin_edges, self.orientation
                )
                self.mesh.set_data(vertices=vertices, faces=faces)

                # Update clim lines and gamma curve with new scaling
                if self._clims is not None:
                    self.set_clims(self._clims, self._gamma, self.log_scale)
