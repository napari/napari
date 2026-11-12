# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "PyYAML>=6.0",
# ]
# ///
"""Check and fix the author order in napari's CITATION.cff.

Usage (``python`` needs PyYAML installed; ``uv run`` in place of ``python``
installs it from the inline script metadata)::

    python tools/check_citation_authors.py check-order
    python tools/check_citation_authors.py write-order

The authors list is split in two by the comment line
``# Community contributors, alphabetized by family-names.``. The core-team
section above it is maintained by hand and kept as is. The community section
below it must be sorted by family-names, then given-names, which
``write-order`` does for you. ``check-order`` also flags duplicate aliases.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from itertools import pairwise
from pathlib import Path

import yaml

CITATION_CFF = Path(__file__).resolve().parent.parent / 'CITATION.cff'
CORE_TEAM_COMMENT = '# Current and emeritus core team members.\n'
COMMUNITY_COMMENT = '# Community contributors, alphabetized by family-names.\n'


def normalize(value: str) -> str:
    """Casefold, collapse whitespace and strip accents, for sorting."""
    value = unicodedata.normalize('NFKD', ' '.join(value.split()).casefold())
    return ''.join(char for char in value if not unicodedata.combining(char))


def sort_key(author: dict) -> tuple[str, ...]:
    return tuple(
        normalize(author.get(key, ''))
        for key in ('family-names', 'given-names', 'alias')
    )


def label(author: dict) -> str:
    return (
        f'{author.get("family-names", "")}, {author.get("given-names", "")} '
        f'({author.get("alias", "")})'
    )


def read_sections(path: Path) -> tuple[str, list[dict], list[dict], str]:
    """Split CITATION.cff into its text before and after the authors list,
    and the core-team and community authors in between."""
    text = path.read_text(encoding='utf-8')
    before, found, rest = text.partition('\nauthors:\n')
    if not found:
        raise ValueError('CITATION.cff has no authors list.')
    lines = rest.splitlines(keepends=True)
    # the authors list ends at the next top-level key
    end = next(
        (i for i, line in enumerate(lines) if line[:1].isalpha()), len(lines)
    )
    authors_text = ''.join(lines[:end])
    core_text, found, community_text = authors_text.partition(
        COMMUNITY_COMMENT
    )
    if not found:
        raise ValueError(
            f'CITATION.cff is missing the line: {COMMUNITY_COMMENT}'
        )
    return (
        before + '\nauthors:\n',
        yaml.safe_load(core_text) or [],
        yaml.safe_load(community_text) or [],
        ''.join(lines[end:]),
    )


def check_order(path: Path) -> int:
    _, core, community, _ = read_sections(path)

    with_alias = [a for a in core + community if a.get('alias')]
    counts = Counter(normalize(a['alias']) for a in with_alias)
    duplicates = [a for a in with_alias if counts[normalize(a['alias'])] > 1]
    if duplicates:
        print(
            'CITATION.cff lists the same alias more than once:',
            file=sys.stderr,
        )
        for author in duplicates:
            print(f'  {label(author)}', file=sys.stderr)
        return 1

    for index, (current, following) in enumerate(pairwise(community), start=1):
        if sort_key(current) > sort_key(following):
            print(
                'Community contributors in CITATION.cff must be alphabetized '
                'by family-names after the core-team section marker.\n'
                f'Out-of-order community pair #{index} and #{index + 1}: '
                f'{label(current)} should sort after {label(following)}.\n'
                'Run `python tools/check_citation_authors.py write-order` '
                'to fix the ordering.',
                file=sys.stderr,
            )
            return 1

    print('CITATION.cff author ordering is valid.')
    return 0


def dump(authors: list[dict]) -> str:
    if not authors:
        return ''
    # never wrap long values (e.g. affiliations) onto continuation lines
    return yaml.safe_dump(
        authors, allow_unicode=True, sort_keys=False, width=sys.maxsize
    )


def write_order(path: Path) -> int:
    before, core, community, after = read_sections(path)
    new_text = (
        before
        + CORE_TEAM_COMMENT
        + dump(core)
        + COMMUNITY_COMMENT
        + dump(sorted(community, key=sort_key))
        + after
    )
    if path.read_text(encoding='utf-8') == new_text:
        print('CITATION.cff author order already matches the project policy.')
    else:
        path.write_text(new_text, encoding='utf-8', newline='\n')
        print(f'Reordered community contributors in {path}.')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--citation-path',
        type=Path,
        default=CITATION_CFF,
        help='Path to the CITATION.cff file to inspect.',
    )
    parser.add_argument('command', choices=['check-order', 'write-order'])
    args = parser.parse_args(argv)
    # print non-ASCII names on Windows consoles
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

    if args.command == 'check-order':
        return check_order(args.citation_path)
    return write_order(args.citation_path)


if __name__ == '__main__':
    raise SystemExit(main())
