import argparse
import os

DEFAULT_NAME = "auto-dependency-upgrades"


def main():
    ref_name = os.environ.get("GITHUB_REF_NAME")
    event = os.environ.get("GITHUB_EVENT_NAME")
    repository = os.environ.get("GITHUB_REPOSITORY")

    parse = argparse.ArgumentParser()
    parse.add_argument("--verbose", action="store_true")
    parse.add_argument("--base_branch", action="store_true")
    args = parse.parse_args()

    if args.verbose:
        print(f"ref_name: {ref_name}")
        print(f"event: {event}")
        print(f"repository: {repository}")
        print(f"base branch: {get_base_branch_name(ref_name, event)}")
        print(f"branch: {get_branch_name(ref_name, event, repository)}")
    elif args.base_branch:
        print(get_base_branch_name(ref_name, event))
    else:
        print(get_branch_name(ref_name, event, repository))


def get_base_branch_name(ref_name, event):
    if ref_name == DEFAULT_NAME:
        return "main"
    if ref_name.startswith(DEFAULT_NAME):
        if event in {"pull_request", "pull_request_target"}:
            return os.environ.get("GITHUB_BASE_REF")
        return ref_name[len(DEFAULT_NAME) + 1 :]
    return ref_name


def get_branch_name(ref_name, event, repository):
    if event == "schedule":
        return DEFAULT_NAME

    if event == "workflow_dispatch":
        if ref_name == "main":
            return DEFAULT_NAME
        return f"{DEFAULT_NAME}-{ref_name}"

    if event == "push":
        if ref_name.startswith(DEFAULT_NAME):
            return ref_name
        if ref_name != "main":
            return f"{DEFAULT_NAME}-{ref_name}"
    if event == "pull_request" and repository == "napari/napari":
        return ref_name

    return "skip"


if __name__ == "__main__":
    main()
