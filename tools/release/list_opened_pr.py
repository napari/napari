import argparse

from release_utils import (
    iter_pull_request,
    setup_cache,
    short_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument('milestone', help='The milestone to list')

args = parser.parse_args()

setup_cache()

pull_list = []

with short_cache(60):
    iterable = iter_pull_request(f'is:pr is:open milestone:{args.milestone}')

for pull in iterable:
    pull_list.append(pull)

print(f"## {len(pull_list)} opened PRs for milestone {args.milestone}")
for pull in pull_list:
    print(f"* [ ] #{pull.number}")
