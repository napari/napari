"""
This script finds all pre-commit PRs that modify not only the pre-commit config
"""
import argparse

from tqdm import tqdm

from release_utils import (
    GH_REPO,
    GH_USER,
    get_commit_counts_from_ancestor,
    get_github,
    get_split_date,
    setup_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')

args = parser.parse_args()

setup_cache()


previous_tag_date = get_split_date(args.from_commit, args.to_commit)

pr_count = get_commit_counts_from_ancestor(args.from_commit, args.to_commit)

pr_to_list = []

for pull_issue in tqdm(
    get_github().search_issues(
        f'repo:{GH_USER}/{GH_REPO} '
        f'merged:>{previous_tag_date.isoformat()} '
        "is:pr "
        'sort:created-asc'
    ),
    desc='Pull Requests...',
    total=pr_count,
):
    pull = pull_issue.as_pull_request()
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
