# Contributing to napari-gui

We welcome your contributions! Please see the provided steps below and never hesitate to contact us.

If you are a new user, we recommend checking out the detailed [Github Guides](https://guides.github.com).

## Setting up a development installation

In order to make changes to `napari-gui`, you will need to [fork](https://guides.github.com/activities/forking/#fork) the
[repository](https://github.com/Napari/napari-gui).

If you are not familiar with `git`, we recommend reading up on [this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Clone the forked repository to your local machine and change directories:
```sh
$ git clone https://github.com/your-username/napari-gui.git
$ cd napari-gui
```

Set the `upstream` remote to the base `napari-gui` repository:
```sh
$ git add upstream https://github.com/Napari/napari-gui.git
```

Install the required dependencies:
```sh
$ pip install -r requirements.txt
```

Make the development version available globally:
```sh
$ pip install -e .
```

## Making changes

Create a new feature branch:
```sh
$ git checkout master -b your-branch-name
```

`git` will automatically detect changes to a repository.
You can view them with:
```sh
$ git status
```

Add and commit your changed files:
```sh
$ git add my-file-or-directory
$ git commit -m "my message"
```

### Help us make sure it's you

Each commit you make must have a [GitHub-registered email](https://github.com/settings/emails)
as the `author`. You can read more [here](https://help.github.com/articles/about-commit-email-addresses/)

To set it, use `git config --global user.email your-address@example.com`.

## Keeping your branches up-to-date

Switch to the `master` branch:
```sh
$ git checkout master
```

Fetch changes and update `master`:
```sh
$ git pull upstream/master
```

This is shorthand for:
```sh
$ git fetch upstream master
$ git merge upstream/master
```

Update your other branches:
```sh
$ git checkout your-branch-name
$ git merge master
```

## Sharing your changes

Update your remote branch:
```sh
$ git push -u origin your-branch-name
```

You can then make a [pull-request](https://guides.github.com/activities/forking/#making-a-pull-request) to `napari-gui`'s `master` branch.

## Questions, comments, and feedback

If you have questions, comments, suggestions for improvement, or any other inquiries
regarding the project, feel free to open an [issue](https://github.com/Napari/napari-gui/issues).

Issues and pull-requests are written in [Markdown](https://guides.github.com/features/mastering-markdown/#what). You can find a comprehensive guide [here](https://guides.github.com/features/mastering-markdown/#syntax).
