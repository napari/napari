import argparse
from datetime import datetime

from tqdm import tqdm

from release_utils import (
    GH_REPO,
    GH_USER,
    get_common_ancestor,
    get_github,
    get_milestone,
    get_repo,
    setup_cache,
)

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')
parser.add_argument(
    "--milestone",
    help="if present then filter issues with a given milestone",
    default=None,
    type=str,
)
parser.add_argument(
    "--skip-triaged",
    action="store_true",
    default=False,
    help="if present then skip triaged PRs",
)
parser.add_argument("--label", help="The label", action="append")

args = parser.parse_args()

if args.label is None:
    args.label = ["bug"]

setup_cache()

repository = get_repo()

if args.milestone is not None:
    if args.mileston.lower() == "none":
        milestone_search_string = "no:milestone"
        milestone = None
    else:
        milestone = get_milestone(args.milestone)
        milestone_search_string = f'milestone:"{milestone.title}"'
else:
    milestone_search_string = ""
    milestone = None


common_ancestor = get_common_ancestor(args.from_commit, args.to_commit)
remote_commit = repository.get_commit(common_ancestor.hexsha)
previous_tag_date = datetime.strptime(
    remote_commit.last_modified, '%a, %d %b %Y %H:%M:%S %Z'
)

probably_solved = repository.get_label("probably solved")
need_to_reproduce = repository.get_label("need to reproduce")

if args.skip_triaged:
    triage_labels = [
        x for x in repository.get_labels() if x.name.startswith("triaged")
    ]
else:
    triage_labels = []

labels = [repository.get_label(label) for label in args.label]

search_string = (
    f'repo:{GH_USER}/{GH_REPO} is:issue is:open '
    f'created:>{previous_tag_date.isoformat()} '
    'sort:updated-desc' + milestone_search_string
)
for label in labels:
    search_string += f' label:"{label.name}"'

iterable = get_github().search_issues(search_string)

issue_list = []

for issue in tqdm(
    iterable,
    desc='issues...',
    total=iterable.totalCount,
):
    if "[test-bot]" in issue.title:
        continue
    if probably_solved in issue.labels:
        continue
    if need_to_reproduce in issue.labels:
        continue
    if args.skip_triaged and any(x in issue.labels for x in triage_labels):
        continue

    issue_list.append(issue)

if len(labels) > 1:
    label_string = "labels " + ", ".join([x.name for x in labels])
else:
    label_string = f"label {labels[0].name}"


header = f"## {len(issue_list)} Opened Issues with {label_string}"

if milestone:
    if milestone_search_string.startswith("no:"):
        header += " and no milestone"
    else:
        header += f" and milestone {milestone.title}"

print(header)

for issue in issue_list:
    print(f" * [ ] #{issue.number}")
