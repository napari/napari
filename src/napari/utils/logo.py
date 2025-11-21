import re
from importlib import resources
from pathlib import Path


def available_logos():
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))
    variants = ['auto']
    for logo in Path(logo_dir).glob('*.svg'):
        if match := re.match(r'(.*)-plain-light', logo.stem):
            variants.append(match.group(1))
    return variants


def get_logo_path(logo, theme):
    logo_dir = Path(resources.files('napari').joinpath('resources', 'logos'))
    if theme not in ('dark', 'light'):
        theme = 'dark'

    if logo == 'auto':
        return logo_dir / f'gradient-plain-{theme}.svg'

    return logo_dir / f'{logo}-plain-{theme}.svg'
