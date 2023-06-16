"""
This is script to cherry pick commits base on PR labels
"""


import argparse
from pathlib import Path

from git import GitCommandError, Repo
from tqdm import tqdm

from release_utils import (
    iter_pull_request,
    pr_num_pattern,
    setup_cache,
    short_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument('base_branch', help='The base branch.')
parser.add_argument('milestone', help='The milestone to list')

LOCAL_DIR = Path(__file__).parent

if not (LOCAL_DIR / "patch_dir").exists():
    (LOCAL_DIR / "patch_dir").mkdir()

args = parser.parse_args()

target_branch = f"v{args.milestone}x"

if not (LOCAL_DIR / "napari_repo").exists():
    repo = Repo.clone_from(
        "git@github.com:napari/napari.git", LOCAL_DIR / "napari_repo"
    )
else:
    repo = Repo(LOCAL_DIR / "napari_repo")

if target_branch not in repo.branches:
    repo.git.checkout(args.base_branch)
    repo.git.checkout('HEAD', b=target_branch)
else:
    repo.git.reset('--hard', "HEAD")
    repo.git.checkout(target_branch)
    # repo.git.pull()

setup_cache()

with short_cache(60):
    iterable = iter_pull_request(f"milestone:{args.milestone} is:merged")

pr_list = sorted(iterable, key=lambda x: x.closed_at)

pr_commits_dict = {}

for commit in repo.iter_commits("main"):
    if (match := pr_num_pattern.search(commit.message)) is not None:
        pr_num = int(match[1])
        pr_commits_dict[pr_num] = commit.hexsha

consumed_pr = set()

for commit in repo.iter_commits(target_branch):
    if (match := pr_num_pattern.search(commit.message)) is not None:
        pr_num = int(match[1])
        consumed_pr.add(pr_num)

for el in pr_list:
    if el.number in consumed_pr:
        print(el, el.number in consumed_pr)


for pull in tqdm(pr_list):
    if pull.number in consumed_pr:
        continue
    # commit = repo.commit(pr_commits_dict[pull.number])
    patch_file = LOCAL_DIR / "patch_dir" / f"{pull.number}.patch"
    if patch_file.exists():
        print(f"Apply patch {patch_file}")
        repo.git.am(str(patch_file))
        continue
    try:
        repo.git.cherry_pick(pr_commits_dict[pull.number])
    except GitCommandError:
        print(pull, pr_commits_dict[pull.number])
        repo.git.mergetool()
        repo.git.cherry_pick('--continue')
        with open(LOCAL_DIR / "patch_dir" / f"{pull.number}.patch", "w") as f:
            f.write(repo.git.format_patch("HEAD~1", '--stdout'))
