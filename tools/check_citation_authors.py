"""Utilities for maintaining napari's CITATION.cff author list.

Usage:
    python tools/check_citation_authors.py check-order
        Validate that community contributors remain alphabetized by
        ``family-names`` after the core-team section marker.

    python tools/check_citation_authors.py write-order
        Rewrite only the community-contributor section into the expected
        alphabetical order. The core-team block is preserved as-is.

    python tools/check_citation_authors.py audit-missing [--repo ...] [--org ...] [--issue-body]
        Compare GitHub contributors across one or more repos (or an entire
        GitHub org) against the ``alias`` values in ``CITATION.cff`` and
        print either the missing usernames or a ready-to-paste issue body
        with ``@mentions``.

Notes:
    - The top block in ``CITATION.cff`` is an explicitly maintained core-team
      section. This script does not alphabetize or otherwise reorder that block.
    - The boundary between the core-team block and the community block is the
      comment line ``# Community contributors, alphabetized by family-names.``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from itertools import pairwise
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CITATION_CFF = REPO_ROOT / 'CITATION.cff'
CORE_TEAM_COMMENT = 'Current and emeritus core team members.'
COMMUNITY_COMMENT = 'Community contributors, alphabetized by family-names.'
KNOWN_BOT_LOGINS = {
    'app/dependabot',
    'dependabot[bot]',
    'github-actions[bot]',
    'napari-bot',
    'pre-commit-ci[bot]',
    'renovate[bot]',
}


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(encoding='utf-8', errors='backslashreplace')


def load_citation_data(path: Path = CITATION_CFF) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def normalize_sort_value(value: str) -> str:
    collapsed = ' '.join(value.split())
    decomposed = unicodedata.normalize('NFKD', collapsed.casefold())
    return ''.join(
        char for char in decomposed if not unicodedata.combining(char)
    )


def author_label(author: dict) -> str:
    given_names = author.get('given-names', '').strip()
    family_names = author.get('family-names', '').strip()
    alias = author.get('alias', '').strip()
    return f'{family_names}, {given_names} ({alias})'


def author_sort_key(author: dict) -> tuple[str, str, str]:
    return (
        normalize_sort_value(author.get('family-names', '')),
        normalize_sort_value(author.get('given-names', '')),
        normalize_sort_value(author.get('alias', '')),
    )


def get_authors(data: dict) -> list[dict]:
    authors = data.get('authors')
    if not isinstance(authors, list):
        raise TypeError('CITATION.cff must contain an authors list.')
    return authors


def check_order(path: Path = CITATION_CFF) -> int:
    _, _, community, _ = split_citation_author_sections(path)
    for index, (current, following) in enumerate(
        pairwise(community),
        start=1,
    ):
        if author_sort_key(current) > author_sort_key(following):
            print(
                'Community contributors in CITATION.cff must be alphabetized by '
                'family-names after the core-team section marker.',
                file=sys.stderr,
            )
            print(
                f'Out-of-order community pair #{index} and #{index + 1}: '
                f'{author_label(current)} should sort after '
                f'{author_label(following)}.',
                file=sys.stderr,
            )
            return 1

    print('CITATION.cff author ordering is valid.')
    return 0


def load_author_from_block(block: str) -> dict:
    payload = yaml.safe_load(block)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            'Each author block in CITATION.cff must contain one author.'
        )
    author = payload[0]
    if not isinstance(author, dict):
        raise TypeError(
            'Each author block in CITATION.cff must parse as a mapping.'
        )
    return author


def _split_author_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current_block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if line.startswith('- '):
            if current_block:
                blocks.append(''.join(current_block))
            current_block = [line]
            continue
        if not current_block:
            raise ValueError(
                'Found non-author content inside the authors section.'
            )
        current_block.append(line)

    if current_block:
        blocks.append(''.join(current_block))
    return blocks


def split_citation_author_sections(
    path: Path,
) -> tuple[list[str], list[dict], list[dict], list[str]]:
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)

    try:
        authors_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith('authors:')
        )
    except StopIteration as error:
        raise ValueError(
            'CITATION.cff does not contain an authors section.'
        ) from error

    suffix_index = len(lines)
    for index in range(authors_index + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith((' ', '-', '\t', '#')):
            suffix_index = index
            break

    prefix_lines = lines[: authors_index + 1]
    author_lines = lines[authors_index + 1 : suffix_index]
    suffix_lines = lines[suffix_index:]

    try:
        community_index = next(
            index
            for index, line in enumerate(author_lines)
            if line.strip() == f'# {COMMUNITY_COMMENT}'
        )
    except StopIteration as error:
        raise ValueError(
            'CITATION.cff must contain a community section marker line: '
            f'# {COMMUNITY_COMMENT}'
        ) from error

    core_lines = author_lines[:community_index]
    community_lines = author_lines[community_index + 1 :]
    core_authors = [
        load_author_from_block(block)
        for block in _split_author_blocks(core_lines)
    ]
    community_authors = [
        load_author_from_block(block)
        for block in _split_author_blocks(community_lines)
    ]
    return prefix_lines, core_authors, community_authors, suffix_lines


def build_author_block(author: dict) -> str:
    return yaml.safe_dump(
        [author],
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def write_order(path: Path = CITATION_CFF) -> int:
    prefix_lines, core_authors, community_authors, suffix_lines = (
        split_citation_author_sections(path)
    )
    sorted_community = sorted(community_authors, key=author_sort_key)
    new_text = (
        ''.join(prefix_lines)
        + f'# {CORE_TEAM_COMMENT}\n'
        + ''.join(build_author_block(author) for author in core_authors)
        + f'# {COMMUNITY_COMMENT}\n'
        + ''.join(build_author_block(author) for author in sorted_community)
        + ''.join(suffix_lines)
    )

    if path.read_text(encoding='utf-8') == new_text:
        print('CITATION.cff author order already matches the project policy.')
        return 0

    path.write_text(new_text, encoding='utf-8')
    print(f'Reordered community contributors in {path}.')
    return 0


def github_login_key(login: str) -> str:
    return normalize_sort_value(login)


def is_bot_login(login: str) -> bool:
    return login in KNOWN_BOT_LOGINS or login.endswith('[bot]')


def fetch_github_contributors(repo: str) -> list[str]:
    result = subprocess.run(
        [
            'gh',
            'api',
            f'repos/{repo}/contributors',
            '--paginate',
            '--slurp',
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or 'Unknown gh api error.'
        raise RuntimeError(f'Failed to fetch GitHub contributors: {stderr}')

    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise TypeError('Unexpected GitHub API response for contributors.')

    seen: set[str] = set()
    contributors: list[str] = []
    for page in payload:
        if not isinstance(page, list):
            raise TypeError(
                'Unexpected GitHub API page while reading contributors.'
            )
        for item in page:
            login = item.get('login') if isinstance(item, dict) else None
            if not isinstance(login, str):
                continue
            if is_bot_login(login):
                continue
            normalized = github_login_key(login)
            if normalized in seen:
                continue
            seen.add(normalized)
            contributors.append(login)
    return contributors


def fetch_org_repos(org: str) -> list[str]:
    """Fetch all non-archived, non-fork repos in a GitHub organization."""
    result = subprocess.run(
        ['gh', 'api', f'/orgs/{org}/repos', '--paginate', '--slurp'],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or 'Unknown gh api error.'
        raise RuntimeError(f'Failed to fetch org repos: {stderr}')

    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise TypeError('Unexpected GitHub API response for org repos.')

    repos: list[str] = []
    for page in payload:
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, dict):
                continue
            if item.get('archived') or item.get('fork'):
                continue
            full_name = item.get('full_name')
            if isinstance(full_name, str):
                repos.append(full_name)
    return sorted(repos)


def missing_contributors(path: Path, repos: list[str]) -> list[str]:
    authors = get_authors(load_citation_data(path))
    aliases = {
        github_login_key(author.get('alias', ''))
        for author in authors
        if isinstance(author.get('alias'), str)
        and author.get('alias', '').strip()
    }
    seen: set[str] = set()
    missing: list[str] = []
    for repo in repos:
        for login in fetch_github_contributors(repo):
            key = github_login_key(login)
            if key in seen:
                continue
            seen.add(key)
            if key not in aliases:
                missing.append(login)
    return missing


def format_issue_body(logins: list[str], repos: list[str]) -> str:
    repos_str = ', '.join(repos)
    mentions = ' '.join(f'@{login}' for login in logins)
    lines = [
        'The following GitHub contributors appear to have commits in '
        f'{repos_str} but do not currently have an `alias` entry in '
        '`CITATION.cff`.',
        '',
        'If you would like to be included, please open a PR that adds your '
        'entry in alphabetical order by `family-names` within the community '
        'contributor section.',
        '',
        mentions or '(No missing contributors found.)',
    ]
    return '\n'.join(lines)


def audit_missing(
    path: Path, repos: list[str], org: str | None, issue_body: bool
) -> int:
    if org:
        org_repos = fetch_org_repos(org)
        # Deduplicate while preserving order: org repos first, then extras
        seen_repos: set[str] = set()
        combined: list[str] = []
        for r in org_repos + repos:
            if r not in seen_repos:
                seen_repos.add(r)
                combined.append(r)
        repos = combined
        print(
            f'Scanning {len(repos)} repos from org "{org}"...',
            file=sys.stderr,
        )

    try:
        logins = missing_contributors(path, repos)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    if issue_body:
        print(format_issue_body(logins, repos))
        return 0

    if not logins:
        print('All GitHub contributors are represented in CITATION.cff.')
        return 0

    print('GitHub contributors missing from CITATION.cff:')
    for login in logins:
        print(login)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--citation-path',
        type=Path,
        default=CITATION_CFF,
        help='Path to the CITATION.cff file to inspect.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('check-order', help='Validate author ordering.')
    subparsers.add_parser(
        'write-order',
        help='Rewrite community contributors into the expected sorted order.',
    )

    audit_parser = subparsers.add_parser(
        'audit-missing',
        help='Report GitHub contributors missing from CITATION.cff aliases.',
    )
    audit_parser.add_argument(
        '--repo',
        action='append',
        dest='repos',
        default=['napari/napari'],
        help='GitHub repository in owner/name format (may be given multiple times).',
    )
    audit_parser.add_argument(
        '--org',
        help=(
            'GitHub organization name; scans all non-archived, non-fork '
            'repos (e.g. --org napari). Can be combined with --repo.'
        ),
    )
    audit_parser.add_argument(
        '--issue-body',
        action='store_true',
        help='Print a ready-to-paste GitHub issue body with @mentions.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'check-order':
        return check_order(args.citation_path)
    if args.command == 'write-order':
        return write_order(args.citation_path)
    if args.command == 'audit-missing':
        return audit_missing(
            args.citation_path, args.repos, args.org, args.issue_body
        )

    parser.error(f'Unsupported command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
