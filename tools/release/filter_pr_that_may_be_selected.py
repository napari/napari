import argparse

from release_utils import (
    get_milestone,
    get_repo,
    get_split_date,
    iter_pull_request,
    setup_cache,
    short_cache,
)

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--skip-triaged",
    action="store_true",
    default=False,
    help="if present then skip triaged PRs",
)
args = parser.parse_args()


setup_cache()

repository = get_repo()

milestone = get_milestone(args.milestone)

label = repository.get_label(args.label) if args.label else None

if args.skip_triaged:
    triage_labels = [
        x for x in repository.get_labels() if x.name.startswith("triaged")
    ]
else:
    triage_labels = []

previous_tag_date = get_split_date(args.from_commit, args.to_commit)

with short_cache(60):
    if milestone is not None:
        iterable = iter_pull_request(f"milestone:{milestone.title} is:merged")
    else:
        iterable = iter_pull_request(
            f'merged:>{previous_tag_date.isoformat()} no:milestone'
        )

pr_to_list = []

for pull in iterable:
    if label is not None and label not in pull.labels:
        continue
    if args.skip_triaged and any(x in pull.labels for x in triage_labels):
        continue
    pr_to_list.append(pull)

if not pr_to_list:
    text = (
        f'## No PRs found with milestone {milestone.title}'
        if milestone
        else '## No PRs found without milestone'
    )
elif milestone:
    text = f'## {len(pr_to_list)} PRs with milestone {milestone.title}'
else:
    text = f'## {len(pr_to_list)} PRs without milestone'

if label:
    text += f' and label {label.name}'

text += ":"
print(text)

for pull in sorted(pr_to_list, key=lambda x: x.closed_at):
    print(f' * [ ] #{pull.number}')
