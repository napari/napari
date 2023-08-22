"""
This file contains the code to create a PR or update an existing one based on the state of the current branch.
"""

import subprocess  # nosec
from contextlib import contextmanager
from os import chdir, environ, getcwd
from pathlib import Path

import requests
from check_updated_packages import get_changed_dependencies

REPO_DIR = Path(__file__).parent.parent
# GitHub API base URL
BASE_URL = "https://api.github.com"


@contextmanager
def cd(path: Path):
    """
    Change directory to the given path and return to the previous one afterwards.
    """
    current_dir = getcwd()
    try:
        chdir(path)
        yield
    finally:
        chdir(current_dir)


def _setup_git_author():
    subprocess.run(
        ["git", "config", "--global", "user.name", "napari-bot"], check=True
    )  # nosec
    subprocess.run(
        [
            "git",
            "config",
            "--global",
            "user.email",
            "napari-bot@users.noreply.github.com",
        ],
        check=True,
    )  # nosec


def create_commit(message: str):
    """
    Create a commit calling git.
    """
    with cd(REPO_DIR):
        subprocess.run(["git", "add", "-u"], check=True)  # nosec
        subprocess.run(["git", "commit", "-m", message], check=True)  # nosec


def push(branch_name: str, update: bool = False):
    """
    Push the current branch to the remote.
    """
    with cd(REPO_DIR):
        if update:
            subprocess.run(["git", "push"], check=True)
        else:
            subprocess.run(
                [
                    "git",
                    "push",
                    "--force",
                    "--set-upstream",
                    "origin",
                    branch_name,
                ],
                check=True,
            )  # nosec


def commit_message(branch_name) -> str:
    changed_direct = get_changed_dependencies(
        all_packages=False, base_branch=branch_name
    )
    if not changed_direct:
        return "Update indirect dependencies"
    return "Update " + ", ".join(f"`{x}`" for x in changed_direct)


def long_description(branch_name: str) -> str:
    all_changed = get_changed_dependencies(
        all_packages=True, base_branch=branch_name
    )
    return "Updated packages: " + ", ".join(f"`{x}`" for x in all_changed)


def create_pr(branch_name: str, access_token: str, repo="napari/napari"):
    """
    Create a PR.
    """
    if branch_name == "main":
        new_branch_name = "auto-update-dependencies"
    else:
        new_branch_name = f"auto-update-dependencies-{branch_name}"

    with cd(REPO_DIR):
        subprocess.run(["git", "checkout", "-B", new_branch_name], check=True)
    create_commit(commit_message(branch_name))

    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # publish the comment
    payload = {
        "title": commit_message(branch_name),
        "body": long_description(branch_name),
        "head": new_branch_name,
        "base": branch_name,
        "maintainer_can_modify": True,
    }
    comment_url = f"{BASE_URL}/repos/{repo}/pulls"
    response = requests.post(comment_url, headers=headers, json=payload)
    response.raise_for_status()


def add_comment_to_pr(
    pull_request_number: int,
    message: str,
    access_token: str,
    repo="napari/napari",
):
    """
    Add a comment to an existing PR.
    """
    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # publish the comment
    payload = {"body": message}
    comment_url = (
        f"{BASE_URL}/repos/{repo}/issues/{pull_request_number}/comments"
    )
    response = requests.post(comment_url, headers=headers, json=payload)
    response.raise_for_status()


def update_pr(branch_name: str, access_token: str):
    """
    Update an existing PR.
    """
    pr_number = get_pr_number()

    create_commit(commit_message(branch_name))
    push(branch_name)
    add_comment_to_pr(
        pr_number,
        long_description(branch_name),
        access_token,
    )


def get_pr_number() -> int:
    """
    Get the PR number from the environment based on the GITHUB_REF variable.

    Returns
    -------
    pr number: int
    """
    github_ref = environ.get("GITHUB_REF")
    refs, pull, number, merge = github_ref.split('/')
    assert refs == 'refs'
    assert pull == 'pull'
    assert merge == 'merge'
    return int(number)


def main():
    branch_name = environ["GITHUB_REF_NAME"]
    event_name = environ["GITHUB_EVENT_NAME"]
    access_token = environ.get("GHA_TOKEN")

    _setup_git_author()

    if event_name in {"schedule", "workflow_dispatch"}:
        create_pr(branch_name)
    elif event_name == "labeled":
        update_pr(branch_name, access_token)


if __name__ == "__main__":
    main()
