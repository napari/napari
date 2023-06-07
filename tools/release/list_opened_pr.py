import argparse

from tqdm import tqdm

from release_utils import (
    GH_REPO,
    GH_USER,
    get_github,
    setup_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument('milestone', help='The milestone to list')

args = parser.parse_args()

setup_cache()

pull_list = []


for pull_issue in tqdm(
    get_github().search_issues(
        f'repo:{GH_USER}/{GH_REPO} is:pr is:open sort:created-asc'
    ),
    desc='Pull Requests...',
):
    pull = pull_issue.as_pull_request()
    if pull.milestone and pull.milestone.title == args.milestone:
        pull_list.append(pull)

print(f"## {len(pull_list)} opened PRs for milestone {args.milestone}")
for pull in pull_list:
    print(f"* [ ] #{pull.number}")
