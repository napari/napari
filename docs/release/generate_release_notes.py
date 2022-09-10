"""Generate the release notes automatically from Github pull requests.
Start with:
```
export GH_TOKEN=<your-gh-api-token>
```
Then, for to include everything from a certain release to main:
```
python /path/to/generate_release_notes.py v0.14.0 main --version 0.15.0
```
Or two include only things between two releases:
```
python /path/to/generate_release_notes.py v.14.2 v0.14.3 --version 0.14.3
```
You should probably redirect the output with:
```
python /path/to/generate_release_notes.py [args] | tee release_notes.md
```
You'll require PyGitHub and tqdm, which you can install with:
```
pip install -e ".[release]"
```
References
https://github.com/scikit-image/scikit-image/blob/main/tools/generate_release_notes.py
https://github.com/scikit-image/scikit-image/issues/3404
https://github.com/scikit-image/scikit-image/issues/3405
"""
import argparse
import os
import re
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from itertools import chain
from os.path import abspath
from pathlib import Path
from warnings import warn

from git import Repo
from github import Github

with suppress(ImportError):
    import requests_cache

    requests_cache.install_cache('github_cache', backend='sqlite')
    # setup cache for speedup execution and reduce number of requests to GitHub API

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    warn(
        'tqdm not installed. This script takes approximately 5 minutes '
        'to run. To view live progressbars, please install tqdm. '
        'Otherwise, be patient.'
    )

    def tqdm(i, **kwargs):
        return i


pr_num_pattern = re.compile(r'\(#(\d+)\)[$\n]')
issue_pattern = re.compile(r'[ ^]#(\d+)')

GH = "https://github.com"
GH_USER = 'napari'
GH_REPO = 'napari'
GH_TOKEN = os.environ.get('GH_TOKEN')
if GH_TOKEN is None:
    raise RuntimeError(
        "It is necessary that the environment variable `GH_TOKEN` "
        "be set to avoid running into problems with rate limiting. "
        "One can be acquired at https://github.com/settings/tokens.\n\n"
        "You do not need to select any permission boxes while generating "
        "the token."
    )

g = Github(GH_TOKEN)
repository = g.get_repo(f'{GH_USER}/{GH_REPO}')

local_repo = Repo(Path(abspath(__file__)).parent.parent.parent)


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')
parser.add_argument(
    '--version', help="Version you're about to release.", default='0.2.0'
)

args = parser.parse_args()

for tag in repository.get_tags():
    if tag.name == args.from_commit:
        previous_tag = tag
        break
else:
    raise RuntimeError(f'Desired tag ({args.from_commit}) not found')

common_ancestor = local_repo.merge_base(args.to_commit, args.from_commit)[0]
remote_commit = repository.get_commit(common_ancestor.hexsha)

# For some reason, go get the github commit from the commit to get
# the correct date
previous_tag_date = datetime.strptime(
    remote_commit.last_modified, '%a, %d %b %Y %H:%M:%S %Z'
)


all_commits = list(
    tqdm(
        repository.get_commits(sha=args.to_commit, since=previous_tag_date),
        desc=f'Getting all commits between {remote_commit.sha} '
        f'and {args.to_commit}',
    )
)
branch_commit = list(
    tqdm(
        repository.get_commits(
            sha=local_repo.tag(args.from_commit).commit.hexsha,
            since=previous_tag_date,
        ),
        desc=f'Getting all commits from release branch {args.from_commit} '
        f'and {remote_commit.sha}',
    )
)
all_hashes = {c.sha for c in all_commits}

consumed_pr = set()

for commit in branch_commit:
    if match := pr_num_pattern.search(commit.commit.message):
        consumed_pr.add(int(match[1]))


def add_to_users(users, new_user):
    if new_user.login in users:
        # reduce obsolete requests to GitHub API
        return
    if new_user.name is None:
        users[new_user.login] = new_user.login
    else:
        users[new_user.login] = new_user.name


authors = set()
committers = set()
reviewers = set()
users = {}

for commit in tqdm(all_commits, desc="Getting committers and authors"):
    if match := pr_num_pattern.search(commit.commit.message):
        if int(match[1]) in consumed_pr:
            continue
            # omit commits from release branch

    if commit.committer is not None:
        add_to_users(users, commit.committer)
        committers.add(commit.committer.login)
    if commit.author is not None:
        add_to_users(users, commit.author)
        authors.add(commit.author.login)

# remove these bots.
committers.discard("web-flow")
authors.discard("azure-pipelines-bot")

highlights = OrderedDict()

highlights['Highlights'] = {}
highlights['New Features'] = {}
highlights['Improvements'] = {}
highlights['Bug Fixes'] = {}
highlights['API Changes'] = {}
highlights['Deprecations'] = {}
highlights['Build Tools'] = {}
other_pull_requests = {}

for pull in tqdm(
    g.search_issues(
        f'repo:{GH_USER}/{GH_REPO} '
        f'merged:>{previous_tag_date.isoformat()} '
        'sort:created-asc'
    ),
    desc='Pull Requests...',
):
    if pull.number in consumed_pr:
        continue
    pr = repository.get_pull(pull.number)
    if pr.merge_commit_sha not in all_hashes:
        continue
    summary = pull.title
    issues_list = []
    if pull.body:
        for x in issue_pattern.findall(pull.body):
            issue = repository.get_issue(int(x))
            if issue.pull_request is None:
                issues_list.append(issue)

    labels = [
        label.name
        for label in chain(pr.labels, *[x.labels for x in issues_list])
    ]

    for review in pr.get_reviews():
        if review.user is not None:
            add_to_users(users, review.user)
            reviewers.add(review.user.login)
    if "bug" in labels:
        highlights['Bug Fixes'][pull.number] = {"summary": summary}
        continue
    if "feature" in labels:
        highlights['New Features'][pull.number] = {"summary": summary}
        continue
    for key, key_dict in highlights.items():
        pr_title_prefix = f'{key}: '.lower()
        if summary.lower().startswith(pr_title_prefix):
            key_dict[pull.number] = {
                'summary': summary[len(pr_title_prefix) :]
            }
            break
    else:
        other_pull_requests[pull.number] = {'summary': summary}


# add Other PRs to the ordered dict to make doc generation easier.
highlights['Other Pull Requests'] = other_pull_requests


# Now generate the release notes
title = f'# napari {args.version}'
print(title)

print(
    f"""
We're happy to announce the release of napari {args.version}!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).
"""
)

print(
    """
For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari
"""
)

for section, pull_request_dicts in highlights.items():
    print(f'## {section}\n')
    for number, pull_request_info in pull_request_dicts.items():
        print(f'- {pull_request_info["summary"]} (#{number})')
    print()


contributors = OrderedDict()

contributors['authors'] = authors
contributors['reviewers'] = reviewers
# ignore committers
# contributors['committers'] = committers

for section_name, contributor_set in contributors.items():
    print()
    if None in contributor_set:
        contributor_set.remove(None)
    committer_str = (
        f'## {len(contributor_set)} {section_name} added to this '
        'release (alphabetical)'
    )
    print(committer_str)
    print()

    for c in sorted(contributor_set, key=lambda x: users[x].lower()):
        commit_link = f"{GH}/{GH_USER}/{GH_REPO}/commits?author={c}"
        print(f"- [{users[c]}]({commit_link}) - @{c}")
    print()
