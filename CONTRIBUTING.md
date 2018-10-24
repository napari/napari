# Contributing to napari-gui

We welcome your contributions! Please see the provided steps below and never hesitate to contact us.

## Steps to contribute

#### First, make sure you get development version of the repository and create your own branch
```
git clone https://github.com/<username>/napari-gui # get the repo
git checkout dev # checkout to dev branch
git branch <branchname> # create your own branch
```

#### Checkout to your own branch for development
- Ensure that your contribution is in its own branch in your fork of the repository (no other changes should be in the branch). You can it with:
```
git checkout BRANCHNAME
git status
```

#### Let us help to make sure it is you
- Each commit in your branch that you are submitting must have your email address as the 'author'. If you need to configure this specifically for your clone of the repository (because you also work on other projects using git), use `git config user.email <address>` in that repository clone to set the address for that single clone.

#### Squash your commits
- Use git rebase to ensure that your contribution applies cleanly to the current HEAD of the master branch in the repository. This also provides a good opportunity to 'squash' any commits in your branch that you'd rather not have live on in infamy!
```
git rebase -i HEAD~3 # if you want to squash three commits from head, you can check your commits with "git log --oneline"
```

#### Make your PR
- Create a GitHub 'pull request' for your branch, targeted at master. See Github Help [link](https://help.github.com/articles/creating-a-pull-request-from-a-fork/), if you need help

## Questions, Comments, and Feedback
If you have questions, comments, suggestions for improvement or any other inquiries regarding our repo, feel free to open an [issue](https://github.com/Napari/napari-gui/issues).

