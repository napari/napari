from datetime import date
from pathlib import Path
from typing import Literal

from napari_resources import logo_path


def _get_seasonal_logo(today: date | None = None, theme: str = 'dark') -> str:
    today = today or date.today()

    # date ranges, adding some buffer around single-day stuff
    ranges = {
        'halloween': ((10, 25), (11, 2)),
        'christmas': ((12, 1), (1, 6)),
        'maythefourth': ((5, 1), (5, 10)),
        'pride': ((6, 21), (7, 4)),  # international pride day 28 june
    }

    theme_variants = {'maythefourth': {'dark': 'sith', 'light': 'jedi'}}

    for name, ((m1, d1), (m2, d2)) in ranges.items():  # noqa: B007
        start = date(today.year, m1, d1)
        end = date(today.year, m2, d2)

        if end < start:
            # rolls over to the next year
            if today >= start or today <= date(today.year, m2, d2):
                break
        else:
            if start <= today <= end:
                break

    else:
        name = 'gradient'

    if name in theme_variants and theme in theme_variants[name]:
        name = theme_variants[name][theme]

    return name


def get_logo_path(
    logo: str,
    template: str,
    theme_type: Literal['dark', 'light'],
    today: date | None = None,
) -> Path:
    if template not in {'plain', 'padded'}:
        raise ValueError('template must be either "plain" or "padded"')

    if logo == 'auto':
        logo = _get_seasonal_logo(today=today, theme=theme_type)

    return logo_path(logo, template, theme_type)
