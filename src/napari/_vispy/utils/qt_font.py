"""Qt-based font rendering for vispy text visuals.

This module provides Qt-based FontManager and TextureFont implementations
that satisfy the vispy API but use Qt font machinery instead of FreeType.
This allows for uniform fonts across the entire napari application.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QFont, QFontMetricsF, QImage, QPainter
from vispy.gloo import TextureAtlas
from vispy.io import load_spatial_filters
from vispy.visuals.text.text import SDFRendererCPU

if TYPE_CHECKING:
    from typing import Any


def _load_glyph_qt(
    qfont: QFont,
    metrics: QFontMetricsF,
    char: str,
    glyphs_dict: dict[str, dict],
) -> None:
    """Load glyph from Qt font into dict.

    This is a Qt-based replacement for vispy's _load_glyph function
    that uses Qt's font rendering instead of FreeType.

    Parameters
    ----------
    qfont : QFont
        Qt font object to use for rendering.
    metrics : QFontMetricsF
        Qt font metrics object.
    char : str
        A single character to be represented.
    glyphs_dict : dict
        Dictionary to store glyph information.
    """

    # Calculate character dimensions and metrics
    bounding_rect = metrics.boundingRect(char)
    advance = metrics.horizontalAdvance(char)

    # Add padding to ensure we capture the full glyph
    padding = 4
    width = int(np.ceil(bounding_rect.width())) + 2 * padding
    height = int(np.ceil(bounding_rect.height())) + 2 * padding

    # Handle empty glyphs (e.g., space)
    if width <= 2 * padding or height <= 2 * padding:
        width = max(width, 2 * padding + 1)
        height = max(height, 2 * padding + 1)
        bitmap = np.zeros((height, width), dtype=np.uint8)
        left = 0
        top = int(metrics.ascent())
    else:
        # Create QImage to render the glyph
        image = QImage(width, height, QImage.Format.Format_Grayscale8)
        image.fill(Qt.GlobalColor.black)

        # Render the character
        painter = QPainter(image)
        painter.setFont(qfont)
        painter.setPen(Qt.GlobalColor.white)

        # Position the character in the image
        # Qt's drawText(x, y, text) places the baseline at y
        # bounding_rect.top() is negative for characters above baseline
        # bounding_rect.bottom() is positive for the bottom edge
        # We need to position so the bounding box is centered with padding
        x_pos = padding - bounding_rect.left()
        # For y: baseline should be at (padding - bounding_rect.top())
        # This puts the top of the bounding box at 'padding' pixels from top
        y_pos = padding - bounding_rect.top()
        painter.drawText(QPointF(x_pos, y_pos), char)
        painter.end()

        # Convert QImage to numpy array
        # QImage has stride/bytesPerLine that may not equal width due to alignment
        ptr = image.constBits()
        if ptr is None:
            raise RuntimeError(
                f'Failed to get image bits for character {char}'
            )
        if hasattr(ptr, 'setsize'):
            ptr.setsize(image.sizeInBytes())

        bytes_per_line = image.bytesPerLine()

        # Create array with proper stride, then extract only the actual image data
        # Type ignore: voidptr is a valid buffer but not recognized by mypy
        full_array = np.frombuffer(ptr, dtype=np.uint8).reshape(  # type: ignore[call-overload]
            (height, bytes_per_line)
        )
        bitmap = full_array[:, :width].copy()

        # Calculate glyph offset (left, top) in FreeType convention
        # left: horizontal offset from pen origin to left edge of bitmap
        # top: vertical offset from baseline to top edge of bitmap (positive = above baseline)
        # Our bitmap includes padding, so we need to account for that
        # The left edge of bitmap is at (bounding_rect.left() - padding)
        # The top edge of bitmap is at (-bounding_rect.top() - padding) from baseline
        left = int(bounding_rect.left() - padding)
        top = int(-bounding_rect.top() + padding)

    # Create glyph dict in vispy format
    kerning_dict: dict[str, float] = {}
    glyph: dict[str, Any] = {
        'char': char,
        'offset': (left, top),
        'bitmap': bitmap,
        'advance': advance,
        'kerning': kerning_dict,
    }
    glyphs_dict[char] = glyph

    # Generate kerning information for all existing glyphs
    # Note: Qt doesn't provide direct kerning access like FreeType,
    # so we approximate it using the difference between advances
    for other_char, other_glyph in glyphs_dict.items():
        if other_char == char:
            continue
        # Qt kerning approximation
        pair_width = metrics.horizontalAdvance(other_char + char)
        expected_width = other_glyph['advance'] + advance
        kerning = pair_width - expected_width
        glyph['kerning'][other_char] = kerning

        # Reverse direction
        pair_width_rev = metrics.horizontalAdvance(char + other_char)
        expected_width_rev = advance + other_glyph['advance']
        kerning_rev = pair_width_rev - expected_width_rev
        other_glyph['kerning'][char] = kerning_rev


class QtTextureFont:
    """Gather a set of glyphs relative to a given font using Qt rendering.

    This is a Qt-based replacement for vispy's TextureFont that uses
    Qt's font rendering machinery instead of FreeType.

    Parameters
    ----------
    font : dict
        Dict with entries "face", "size", "bold", "italic".
    renderer : instance of SDFRenderer
        SDF renderer to use.
    """

    def __init__(self, font: dict[str, Any], renderer: SDFRendererCPU) -> None:
        self._atlas = TextureAtlas(dtype=np.uint8)
        self._atlas.wrapping = 'clamp_to_edge'
        self._kernel, _ = load_spatial_filters()
        self._renderer = renderer
        self._font = deepcopy(font)
        self._font['size'] = 256  # use high resolution point size for SDF
        self._lowres_size = 64  # end at this point size for storage
        assert (self._font['size'] % self._lowres_size) == 0
        # spread/border at the high-res for SDF calculation
        self._spread = 32
        assert self._spread % self.ratio == 0
        self._glyphs: dict[str, dict[str, Any]] = {}

        # Create and cache Qt font and metrics objects
        self._qfont = QFont(self._font['face'], self._font['size'])
        self._qfont.setBold(self._font.get('bold', False))
        self._qfont.setItalic(self._font.get('italic', False))
        self._metrics = QFontMetricsF(self._qfont)

    @property
    def ratio(self) -> int:
        """Ratio of the initial high-res to final stored low-res glyph."""
        return self._font['size'] // self._lowres_size

    @property
    def slop(self) -> int:
        """Extra space along each glyph edge due to SDF borders."""
        return self._spread // self.ratio

    def __getitem__(self, char: str) -> dict[str, Any]:
        if not (isinstance(char, str) and len(char) == 1):
            raise TypeError('index must be a 1-character string')
        if char not in self._glyphs:
            self._load_char(char)
        return self._glyphs[char]

    def _load_char(self, char: str) -> None:
        """Build and store a glyph corresponding to an individual character.

        Parameters
        ----------
        char : str
            A single character to be represented.
        """
        assert isinstance(char, str)
        assert len(char) == 1
        assert char not in self._glyphs

        # Load new glyph data using Qt
        _load_glyph_qt(self._qfont, self._metrics, char, self._glyphs)

        # Put new glyph into the texture
        glyph = self._glyphs[char]
        bitmap = glyph['bitmap']

        # Convert to padded array
        data = np.zeros(
            (
                bitmap.shape[0] + 2 * self._spread,
                bitmap.shape[1] + 2 * self._spread,
            ),
            np.uint8,
        )
        data[self._spread : -self._spread, self._spread : -self._spread] = (
            bitmap
        )

        # Store, while scaling down to proper size
        height = data.shape[0] // self.ratio
        width = data.shape[1] // self.ratio
        region = self._atlas.get_free_region(width + 2, height + 2)
        if region is None:
            raise RuntimeError('Cannot store glyph')
        x, y, w, h = region
        x, y, w, h = x + 1, y + 1, w - 2, h - 2

        self._renderer.render_to_texture(data, self._atlas, (x, y), (w, h))
        u0 = x / float(self._atlas.shape[1])
        v0 = y / float(self._atlas.shape[0])
        u1 = (x + w) / float(self._atlas.shape[1])
        v1 = (y + h) / float(self._atlas.shape[0])
        texcoords = (u0, v0, u1, v1)
        glyph.update({'size': (w, h), 'texcoords': texcoords})


class QtFontManager:
    """Helper to create QtTextureFont instances and reuse them when possible.

    This is a Qt-based replacement for vispy's FontManager that uses
    Qt's font rendering machinery.

    Parameters
    ----------
    method : str, optional
        Rendering method ('cpu' or 'gpu'). Default is 'cpu'.
        Currently only 'cpu' is supported for Qt-based rendering.
    """

    def __init__(self, method: str = 'cpu') -> None:
        self._fonts: dict[str, QtTextureFont] = {}
        if not isinstance(method, str) or method not in ('cpu', 'gpu'):
            raise ValueError(
                f'method must be "cpu" or "gpu", got {method} ({type(method)})'
            )
        if method == 'cpu':
            self._renderer = SDFRendererCPU()
        else:  # method == 'gpu':
            from vispy.visuals.text.text import SDFRendererGPU

            self._renderer = SDFRendererGPU()

    def get_font(
        self, face: str, bold: bool = False, italic: bool = False
    ) -> QtTextureFont:
        """Get a font described by face and size.

        Parameters
        ----------
        face : str
            Font face name.
        bold : bool, optional
            Whether to use bold weight.
        italic : bool, optional
            Whether to use italic style.

        Returns
        -------
        QtTextureFont
            The texture font instance.
        """
        key = f'{face}-{bold}-{italic}'
        if key not in self._fonts:
            font = {'face': face, 'bold': bold, 'italic': italic}
            self._fonts[key] = QtTextureFont(font, self._renderer)
        return self._fonts[key]
