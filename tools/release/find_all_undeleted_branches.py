import argparse
from contextlib import suppress

from github.GithubException import GithubException
from tqdm import tqdm

from release_utils import GH_REPO, GH_USER, get_github, setup_cache

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('user_name', help='name to search fr undeleted branches')
args = parser.parse_args()

setup_cache()

user = get_github().get_user(args.user_name)

pull_requests = get_github().search_issues(
    f'repo:{GH_USER}/{GH_REPO} '
    'is:closed '
    "is:pr "
    f"author:{user.login} "
    'sort:created-asc'
)

to_remove_branches = []

for pull_issue in tqdm(pull_requests):
    pull = pull_issue.as_pull_request()
    with suppress(GithubException):
        pull.head.repo.get_branch(pull.head.ref)
        to_remove_branches.append(pull)

if not to_remove_branches:
    print("No undeleted branches found")
    exit(0)

print(f"Found {len(to_remove_branches)} undeleted branches")
for pull in to_remove_branches:
    print(f" {pull.title} {pull.html_url}")
