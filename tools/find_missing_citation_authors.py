# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "PyYAML>=6.0",
# ]
# ///
"""Find GitHub contributors who are missing from napari's CITATION.cff.

Needs an authenticated GitHub CLI (``gh``). Usage (``uv run`` installs
PyYAML from the inline script metadata)::

    uv run tools/find_missing_citation_authors.py [--repo OWNER/NAME ...]
        [--org ORG] [--issue-body]

Contributors are matched case-insensitively against the ``alias`` values
in CITATION.cff. Without ``--repo`` or ``--org``, napari/napari is checked.
``--issue-body`` prints a ready-to-paste issue body with ``@mentions``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

CITATION_CFF = Path(__file__).resolve().parent.parent / 'CITATION.cff'
DEFAULT_REPO = 'napari/napari'
# bots whose login doesn't end in [bot]
BOT_LOGINS = {'app/dependabot', 'napari-bot'}


def gh(*args: str) -> str:
    result = subprocess.run(
        ['gh', *args], capture_output=True, text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        sys.exit(f'`gh {" ".join(args)}` failed: {result.stderr.strip()}')
    return result.stdout


def gh_api_list(endpoint: str) -> list[dict]:
    """Return all items of a paginated GitHub API list endpoint."""
    pages = json.loads(gh('api', endpoint, '--paginate', '--slurp'))
    return [item for page in pages for item in page]


def org_repos(org: str) -> list[str]:
    """Return the non-archived, non-fork repos of a GitHub organization."""
    return sorted(
        repo['full_name']
        for repo in gh_api_list(f'orgs/{org}/repos')
        if not repo['archived'] and not repo['fork']
    )


def contributors(repo: str) -> list[str]:
    return [
        user['login'] for user in gh_api_list(f'repos/{repo}/contributors')
    ]


def is_bot(login: str) -> bool:
    return login in BOT_LOGINS or login.endswith('[bot]')


def missing_logins(citation_path: Path, repos: list[str]) -> list[str]:
    citation = yaml.safe_load(citation_path.read_text(encoding='utf-8'))
    aliases = {
        author['alias'].strip().casefold()
        for author in citation['authors']
        if author.get('alias')
    }
    missing: dict[str, str] = {}
    for repo in repos:
        for login in contributors(repo):
            if not is_bot(login) and login.casefold() not in aliases:
                missing.setdefault(login.casefold(), login)
    return list(missing.values())


def format_issue_body(logins: list[str], repos: list[str]) -> str:
    mentions = ' '.join(f'@{login}' for login in logins)
    return (
        'The following GitHub contributors appear to have commits in '
        f'{", ".join(repos)} but do not currently have an `alias` entry in '
        '`CITATION.cff`.\n\n'
        'If you would like to be included, please open a PR that adds your '
        'entry in alphabetical order by `family-names` within the community '
        'contributor section.\n\n'
        f'{mentions or "(No missing contributors found.)"}'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--citation-path',
        type=Path,
        default=CITATION_CFF,
        help='Path to the CITATION.cff file to inspect.',
    )
    parser.add_argument(
        '--repo',
        action='append',
        dest='repos',
        default=[],
        help='GitHub repository as OWNER/NAME; may be given multiple times.',
    )
    parser.add_argument(
        '--org',
        help='GitHub organization; checks all its non-archived, non-fork '
        'repos. Can be combined with --repo.',
    )
    parser.add_argument(
        '--issue-body',
        action='store_true',
        help='Print a ready-to-paste GitHub issue body with @mentions.',
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8')

    repos = args.repos
    if args.org:
        repos = list(dict.fromkeys(org_repos(args.org) + repos))
        print(
            f'Scanning {len(repos)} repos from {args.org}...', file=sys.stderr
        )
    repos = repos or [DEFAULT_REPO]

    logins = missing_logins(args.citation_path, repos)
    if args.issue_body:
        print(format_issue_body(logins, repos))
        return 0
    if not logins:
        print('All GitHub contributors are represented in CITATION.cff.')
        return 0
    print('GitHub contributors missing from CITATION.cff:')
    print('\n'.join(logins))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
