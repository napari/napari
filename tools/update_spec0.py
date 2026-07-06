#!/usr/bin/env python3
"""Update pyproject.toml dependency minimums per SPEC 0 schedule.

Usage:
    uvx --with git+https://github.com/scientific-python/spec0-action.git -- python tools/update_spec0.py
    uvx --with git+https://github.com/scientific-python/spec0-action.git -- python tools/update_spec0.py --dry-run
    uvx --with git+https://github.com/scientific-python/spec0-action.git -- python tools/update_spec0.py --update-all 2
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from spec0_action import (
    read_schedule,
    read_toml,
    update_pyproject_toml,
    write_toml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / 'pyproject.toml'
SCHEDULE_URL = (
    'https://github.com/scientific-python/'
    'spec0-action/releases/latest/download/schedule.json'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing',
    )
    parser.add_argument(
        '--update-all',
        type=float,
        default=0,
        help='Also bump non-SPEC0 deps (default: 0 = skip)',
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    if not PYPROJECT.exists():
        print(
            f'No pyproject.toml found at {PYPROJECT}',
            file=__import__('sys').stderr,
        )
        __import__('sys').exit(1)

    # Download schedule
    print('Downloading SPEC 0 schedule...')
    resp = urllib.request.urlopen(SCHEDULE_URL, timeout=30)
    schedule_data = json.loads(resp.read().decode())
    print(f'  {len(schedule_data)} quarterly entries loaded')

    # Write schedule to temp file (read_schedule expects a file path)
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(schedule_data, f)
        sched_path = f.name

    try:
        d = read_toml(str(PYPROJECT))
        s = read_schedule(sched_path)
        kwargs = {}
        if args.update_all > 0:
            kwargs['update_all'] = args.update_all
        update_pyproject_toml(d, s, **kwargs)

        if args.dry_run:
            o = json.dumps(read_toml(str(PYPROJECT)), indent=2, sort_keys=True)
            n = json.dumps(d, indent=2, sort_keys=True)
            if o != n:
                import difflib

                for line in difflib.unified_diff(
                    o.splitlines(),
                    n.splitlines(),
                    fromfile='original',
                    tofile='updated',
                    lineterm='',
                ):
                    print(line)
            else:
                print('No changes')
        else:
            write_toml(str(PYPROJECT), d)
            print('Written to pyproject.toml')
    finally:
        os.unlink(sched_path)


if __name__ == '__main__':
    main()
