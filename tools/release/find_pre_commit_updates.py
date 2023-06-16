"""
This script finds all pre-commit PRs that modify not only the pre-commit config
"""
import argparse

from release_utils import (
    get_split_date,
    iter_pull_request,
    setup_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')

args = parser.parse_args()

setup_cache()


previous_tag_date = get_split_date(args.from_commit, args.to_commit)

pr_to_list = []

for pull in iter_pull_request(f'merged:>{previous_tag_date.isoformat()} '):
    if "[pre-commit.ci]" in pull.title and pull.changed_files > 1:
        pr_to_list.append(pull)
    # find PR without milestone
    # if "[pre-commit.ci]" in pull.title and pull.milestone is None:
    #     pr_to_list.append(pull)


if not pr_to_list:
    print('No PRs found')
    exit(0)


for pull in sorted(pr_to_list, key=lambda x: x.closed_at):
    print(f' * [ ] #{pull.number} {pull.html_url} {pull.milestone}')
