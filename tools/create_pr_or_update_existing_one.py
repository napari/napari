"""
This file contains the code to create a PR or update an existing one based on the state of the current branch.
"""

import logging
import os
import subprocess  # nosec
from contextlib import contextmanager
from os import chdir, environ, getcwd
from pathlib import Path

import requests
from check_updated_packages import get_changed_dependencies

REPO_DIR = Path(__file__).parent.parent / "napari_repo"
# GitHub API base URL
BASE_URL = "https://api.github.com"
DEFAULT_BRANCH_NAME = "auto-update-dependencies"


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


def create_commit(message: str, branch_name: str = ""):
    """
    Create a commit calling git.
    """
    with cd(REPO_DIR):
        if branch_name:
            subprocess.run(["git", "checkout", "-B", branch_name], check=True)
        subprocess.run(["git", "add", "-u"], check=True)  # nosec
        subprocess.run(["git", "commit", "-m", message], check=True)  # nosec


def push(branch_name: str, update: bool = False):
    """
    Push the current branch to the remote.
    """
    with cd(REPO_DIR):
        logging.info("go to dir %s", REPO_DIR)
        if update:
            logging.info("Pushing to %s", branch_name)
            subprocess.run(
                [
                    "git",
                    "push",
                    "--force",
                    "--set-upstream",
                    "napari-bot",
                    branch_name,
                ],
                check=True,
                capture_output=True,
            )
        else:
            logging.info("Force pushing to %s", branch_name)
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
                capture_output=True,
            )  # nosec


def commit_message(branch_name) -> str:
    with cd(REPO_DIR):
        changed_direct = get_changed_dependencies(
            all_packages=False,
            base_branch=branch_name,
            python_version="3.11",
            src_dir=REPO_DIR,
        )
    if not changed_direct:
        return "Update indirect dependencies"
    return "Update " + ", ".join(f"`{x}`" for x in changed_direct)


def long_description(branch_name: str) -> str:
    with cd(REPO_DIR):
        all_changed = get_changed_dependencies(
            all_packages=True,
            base_branch=branch_name,
            python_version="3.11",
            src_dir=REPO_DIR,
        )
    return "Updated packages: " + ", ".join(f"`{x}`" for x in all_changed)


def create_pr_with_push(branch_name: str, access_token: str, repo=""):
    """
    Create a PR.
    """
    if branch_name == "main":
        new_branch_name = DEFAULT_BRANCH_NAME
    else:
        new_branch_name = f"{DEFAULT_BRANCH_NAME}-{branch_name}"

    if not repo:
        repo = os.environ.get("GITHUB_REPOSITORY", "napari/napari")
    with cd(REPO_DIR):
        subprocess.run(["git", "checkout", "-B", new_branch_name], check=True)
    create_commit(commit_message(branch_name))
    push(new_branch_name)

    logging.info("Create PR for branch %s", new_branch_name)
    if pr_number := list_pr_for_branch(
        new_branch_name, access_token, repo=repo
    ):
        update_own_pr(pr_number, access_token, branch_name, repo)
    else:
        create_pr(
            base_branch=branch_name,
            new_branch=new_branch_name,
            access_token=access_token,
            repo=repo,
        )


def update_own_pr(pr_number: int, access_token: str, base_branch: str, repo):
    headers = {"Authorization": f"token {access_token}"}
    payload = {
        "title": commit_message(base_branch),
        "body": long_description(base_branch),
    }
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}"
    logging.info("Update PR with payload: %s in %s", str(payload), url)
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()


def list_pr_for_branch(branch_name: str, access_token: str, repo=""):
    """
    check if PR for branch exists
    """
    org_name = repo.split('/')[0]
    url = f"{BASE_URL}/repos/{repo}/pulls?state=open&head={org_name}:{branch_name}"
    response = requests.get(url)
    response.raise_for_status()
    if response.json():
        return response.json()[0]['number']
    return None


def create_pr(
    base_branch: str, new_branch: str, access_token: str, repo, source_user=""
):
    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # publish the comment
    payload = {
        "title": commit_message(base_branch),
        "body": long_description(base_branch),
        "head": new_branch,
        "base": base_branch,
        "maintainer_can_modify": True,
    }
    if source_user:
        payload["head"] = f"{source_user}:{new_branch}"
    pull_request_url = f"{BASE_URL}/repos/{repo}/pulls"
    logging.info(
        "Create PR with payload: %s in %s", str(payload), pull_request_url
    )
    response = requests.post(pull_request_url, headers=headers, json=payload)
    response.raise_for_status()
    logging.info("PR created: %s", response.json()["html_url"])
    add_label(repo, response.json()["number"], "maintenance", access_token)


def add_label(repo, pr_num, label, access_token):
    pull_request_url = f"{BASE_URL}/repos/{repo}/issues/{pr_num}/labels"
    headers = {"Authorization": f"token {access_token}"}
    payload = {"labels": [label]}

    logging.info("Add labels: %s in %s", str(payload), pull_request_url)
    response = requests.post(pull_request_url, headers=headers, json=payload)
    response.raise_for_status()
    logging.info("Labels added: %s", response.json())


def add_comment_to_pr(
    pull_request_number: int,
    message: str,
    repo="napari/napari",
):
    """
    Add a comment to an existing PR.
    """
    # Prepare the headers with the access token
    headers = {"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"}

    # publish the comment
    payload = {"body": message}
    comment_url = (
        f"{BASE_URL}/repos/{repo}/issues/{pull_request_number}/comments"
    )
    response = requests.post(comment_url, headers=headers, json=payload)
    response.raise_for_status()


def update_pr(branch_name: str):
    """
    Update an existing PR.
    """
    pr_number = get_pr_number()

    target_repo = os.environ.get('FULL_NAME')

    new_branch_name = f"auto-update-dependencies/{target_repo}/{branch_name}"

    if (
        target_repo == os.environ.get("GITHUB_REPOSITORY", "napari/napari")
        and branch_name == DEFAULT_BRANCH_NAME
    ):
        new_branch_name = DEFAULT_BRANCH_NAME

    create_commit(commit_message(branch_name), branch_name=new_branch_name)
    comment_content = long_description(f"origin/{branch_name}")

    try:
        push(new_branch_name, update=branch_name != DEFAULT_BRANCH_NAME)
    except subprocess.CalledProcessError as e:
        if "create or update workflow" in e.stderr.decode():
            logging.info("Workflow file changed. Skip PR create.")
            comment_content += (
                "\n\n This PR contains changes to the workflow file. "
            )
            comment_content += "Please download the artifact and update the constraints files manually. "
            comment_content += f"Artifact: https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'napari/napari')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
        else:
            raise
    else:
        if new_branch_name != DEFAULT_BRANCH_NAME:
            comment_content += update_external_pr_comment(
                target_repo, branch_name, new_branch_name
            )

    add_comment_to_pr(
        pr_number,
        comment_content,
        repo=os.environ.get("GITHUB_REPOSITORY", "napari/napari"),
    )
    logging.info("PR updated: %s", pr_number)


def update_external_pr_comment(
    target_repo: str, branch_name: str, new_branch_name: str
) -> str:
    comment = "\n\nThis workflow cannot automatically update your PR or create PR to your repository. "
    comment += "But you could open such PR by clicking the link: "
    comment += f"https://github.com/{target_repo}/compare/{branch_name}...napari-bot:{new_branch_name}."
    comment += "\n\n"
    comment += "You could also get the updated files from the "
    comment += f"https://github.com/napari-bot/napari/tree/{new_branch_name}/resources/constraints. "
    comment += "Or ask the maintainers to provide you the contents of the constraints artifact "
    comment += f"from the run https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'napari/napari')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
    return comment


def get_pr_number() -> int:
    """
    Get the PR number from the environment based on the PR_NUMBER variable.

    Returns
    -------
    pr number: int
    """
    pr_number = environ.get("PR_NUMBER")
    logging.info("PR_NUMBER: %s", pr_number)
    return int(pr_number)


def main():
    event_name = environ.get("GITHUB_EVENT_NAME")
    branch_name = environ.get("BRANCH")

    access_token = environ.get("GHA_TOKEN_MAIN_REPO")

    _setup_git_author()

    logging.basicConfig(level=logging.INFO)
    logging.info("Branch name: %s", branch_name)
    logging.info("Event name: %s", event_name)

    if event_name in {"schedule", "workflow_dispatch"}:
        logging.info("Creating PR")
        create_pr_with_push(branch_name, access_token)
    elif event_name == "issue_comment":
        logging.info("Updating PR")
        update_pr(branch_name)
    elif event_name == "pull_request":
        logging.info(
            "Pull request run. We cannot add comment or create PR. Please download the artifact."
        )
    else:
        raise ValueError(f"Unknown event name: {event_name}")


if __name__ == "__main__":
    main()
