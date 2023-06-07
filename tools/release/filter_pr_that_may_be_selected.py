import argparse
from datetime import datetime

from release_utils import (
    GH_REPO,
    GH_USER,
    get_commit_counts_from_ancestor,
    get_common_ancestor,
    get_github,
    get_milestone,
    get_repo,
    setup_cache,
)
from tqdm import tqdm

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')
parser.add_argument(
    "--milestone",
    help="if present then filter PR with a given milestone",
    default=None,
    type=str,
)
parser.add_argument(
    "--label",
    help="if present then filter PR with a given label",
    default=None,
    type=str,
)
args = parser.parse_args()


setup_cache()

repository = get_repo()

milestone = get_milestone(args.milestone)

label = repository.get_label(args.label) if args.label else None

common_ancestor = get_common_ancestor(args.from_commit, args.to_commit)
remote_commit = repository.get_commit(common_ancestor.hexsha)
previous_tag_date = datetime.strptime(
    remote_commit.last_modified, '%a, %d %b %Y %H:%M:%S %Z'
)

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
    if milestone is not None and pull.milestone != milestone:
        continue
    if milestone is None and (pull.milestone or not pull.merged):
        continue
    if label is not None and label not in pull.labels:
        continue
    pr_to_list.append(pull)

if not pr_to_list:
    print('No PRs found')
    exit(0)


if milestone:
    text = f'## PRs with milestone {milestone.title}'
else:
    text = '## PRs without milestone'

if label:
    text += f' and label {label.name}'

text += ":"
print(text)

for pull in pr_to_list:
    print(f'* [ ] #{pull.number}')
