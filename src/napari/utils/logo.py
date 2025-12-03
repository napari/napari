import re
from datetime import date
from importlib import resources
from pathlib import Path


def available_logos():
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))
    variants = ['auto']
    for logo in Path(logo_dir).glob('*.svg'):
        if match := re.match(r'(.*)-plain-light', logo.stem):
            variants.append(match.group(1))
    return sorted(variants) 


def _get_seasonal_logo(today=None, theme='dark'):
    today = today or date.today()

    ranges = {
        'halloween': ((10, 25), (11, 2)),
        'christmas': ((12, 1), (1, 6)),
        'maythefourth': ((5, 1), (5, 10)),  # let's give it some leeway :P
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

    if name in theme_variants:
        name = theme_variants[name].get(theme)

    return name


def get_logo_path(logo, theme, today=None):
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))
    if theme not in {'dark', 'light'}:
        theme = 'dark'

    if logo == 'auto':
        logo = _get_seasonal_logo(today=today, theme=theme)

    return logo_dir / f'{logo}-plain-{theme}.svg'
