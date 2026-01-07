from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path


@lru_cache
def available_logos() -> list[str]:
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))  # type: ignore
    variants = ['auto']
    for logo in Path(logo_dir).glob('*-plain-light.svg'):
        variants.append(logo.stem.rsplit('-', 2)[0])
    return sorted(variants)


def _get_seasonal_logo(today: date | None = None, theme: str = 'dark') -> str:
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

    if name in theme_variants and theme in theme_variants[name]:
        name = theme_variants[name][theme]

    return name


def get_logo_path(logo: str, theme: str, today: date | None = None) -> Path:
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))  # type: ignore
    # eventually we should actually use the dark/light "type" that the theme spec of npe2 allows,
    # which is currently unused in napari
    if theme not in {'dark', 'light'}:
        theme = 'dark'

    if logo == 'auto':
        logo = _get_seasonal_logo(today=today, theme=theme)

    return logo_dir / f'{logo}-plain-{theme}.svg'
