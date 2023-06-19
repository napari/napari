import contextlib
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from github import Github, Milestone
from tqdm import tqdm

pr_num_pattern = re.compile(r'\(#(\d+)\)(?:$|\n)')


@contextmanager
def short_cache(new_time):
    """
    context manager for change cache time in requests_cache
    """
    try:
        import requests_cache
    except ImportError:
        yield
        return

    import requests

    if requests_cache.get_cache() is None:
        yield
        return
    session = requests.Session()
    old_time = session.expire_after
    session.expire_after = new_time
    try:
        yield
    finally:
        session.expire_after = old_time


def setup_cache(timeout=3600):
    """
    setup cache for speedup execution and reduce number of requests to GitHub API
    by default cache will expire after 1h (3600s)
    """
    try:
        import requests_cache
    except ImportError:
        print("requests_cache not installed", file=sys.stderr)
        return

    """setup cache for requests"""
    requests_cache.install_cache(
        'github_cache', backend='sqlite', expire_after=timeout
    )


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

_G = None


def get_github():
    global _G
    if _G is None:
        _G = Github(GH_TOKEN)
    return _G


def get_repo():
    g = get_github()
    return g.get_repo(f'{GH_USER}/{GH_REPO}')


def get_local_repo(path=None):
    """
    get local repository
    """
    from pathlib import Path

    from git import Repo

    if path is not None:
        return Repo(path)
    return Repo(Path(__file__).parent.parent.parent)


def get_common_ancestor(commit1, commit2):
    """
    find common ancestor for two commits
    """
    local_repo = get_local_repo()
    return local_repo.merge_base(commit1, commit2)[0]


def get_commits_to_ancestor(ancestor, rev="main"):
    local_repo = get_local_repo()
    yield from local_repo.iter_commits(f'{ancestor.hexsha}..{rev}')


def get_commit_counts_from_ancestor(release, rev="main"):
    """
    get number of commits from ancestor to release
    """
    ancestor = get_common_ancestor(release, rev)
    return sum(
        pr_num_pattern.search(c.message) is not None
        for c in get_commits_to_ancestor(ancestor, rev)
    )


def get_milestone(
    milestone_name: Optional[str],
) -> Optional[Milestone.Milestone]:
    if milestone_name is None:
        return None
    repository = get_repo()
    with contextlib.suppress(ValueError):
        return repository.get_milestone(int(milestone_name))

    for milestone in repository.get_milestones():
        if milestone.title == milestone_name:
            return milestone
    raise RuntimeError(f'Milestone {milestone_name} not found')


def get_split_date(previous_release, rev="main"):
    common_ancestor = get_common_ancestor(previous_release, rev)
    remote_commit = get_repo().get_commit(common_ancestor.hexsha)
    return datetime.strptime(
        remote_commit.last_modified, '%a, %d %b %Y %H:%M:%S %Z'
    )


def iter_pull_request(additional_query):
    iterable = get_github().search_issues(
        f"repo:{GH_USER}/{GH_REPO} "
        "is:pr "
        "sort:created-asc " + additional_query
    )
    print(
        f"Found {iterable.totalCount} pull requests on query: {additional_query}",
        file=sys.stderr,
    )
    for pull_issue in tqdm(
        iterable,
        desc='Pull Requests...',
        total=iterable.totalCount,
    ):
        yield pull_issue.as_pull_request()
