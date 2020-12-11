from typing import Dict

from ..utils.events.dataclass import evented_dataclass
from ..utils.theme import palettes

DEFAULT_THEME = 'dark'
CUSTOM_THEME = 'custom'


@evented_dataclass
class Style:
    """Style object with style information for the viewer.

    Parameters
    ----------
    pallete : dict
        Color pallete for the viewer
    theme : str
        Theme for the viewer.

    Attributes
    ----------
    available_themes : dict of str: dict of str: str
        Available palettes indexed by theme.
    pallete : dict of str: str
        Color pallete for the viewer
    theme : str
        Theme for the viewer.
    """

    palette: Dict[str, str] = None
    theme: str = DEFAULT_THEME

    def __post_init__(self):
        self._on_theme_set(self.theme)

    def _on_theme_set(self, theme):
        """Update the palette based on the theme."""
        if theme in palettes:
            self._palette = self.available_themes[theme]
        else:
            raise ValueError(
                f"Theme '{theme}' not found; "
                f"options are {list(self.available_themes)}."
            )

    def _on_pallet_set(self, palette):
        """Set the theme to be custom."""
        # If pallete is set directly then theme is set to be custom
        for existing_theme, existing_palette in self.available_themes.items():
            if existing_palette == palette:
                self._theme = existing_theme
                return
        self._theme = CUSTOM_THEME

    @property
    def available_themes(self):
        """dict of str: dict of str: str. Preset color palettes, indexed by theme."""
        return palettes

    def _advance_theme(self):
        """Advance theme to next theme in list of themes."""
        themes = list(self.available_themes)
        if self.theme in themes:
            new_theme_index = (themes.index(self.theme) + 1) % len(themes)
        else:
            new_theme_index = 0
        self.theme = themes[new_theme_index]
