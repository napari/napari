(napari-contributing)=
# Contributing guide

We welcome your contributions! Please see the provided steps below and never hesitate to contact us.

If you are a new user, we recommend checking out the detailed [Github Docs](https://docs.github.com/en).

(dev-installation)=
## Setting up a development installation

In order to make changes to `napari`, you will need to [fork](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#forking-a-repository) the
[repository](https://github.com/napari/napari). If you are not familiar with `git`, we recommend reading up on [this guide](https://docs.github.com/en/get-started/using-git/about-git#basic-git-commands).

1. Clone the forked repository to your local machine and change directories:

    ```sh
    git clone https://github.com/your-username/napari.git
    cd napari
    ```

2. Set the `upstream` remote to the base `napari` repository:

    ```sh
    git remote add upstream https://github.com/napari/napari.git
    ```

3. If you haven't already, create a development environment:

    ::::{tab-set}

    :::{tab-item} Using `conda`
    After [installing `conda`](https://www.anaconda.com/products/distribution), create an environment called `napari-env` with Python {{ python_version }} and activate it.

    {{ conda_create_env }}
    :::

    :::{tab-item} Using `venv`
    After installing Python on your machine, create a virtual environment on your terminal and activate it. On Linux and MacOS, you can run
    ```sh
    python -m venv <path-to-env>
    source <path-to-env>/bin/activate
    ```
    See the [venv](https://docs.python.org/3/library/venv.html) documentation for instructions on Windows.
    :::

    ::::

    ```{note}
    It is highly recommended to create a fresh environment when working with
    napari, to prevent issues with outdated or conflicting packages in your
    development environment.
    ```

4. Install the package in editable mode, along with all of the developer tools.

    If you only want to use napari, you can install it on most macOS, Linux and
    Windows systems with Python {{ python_version_range }}
    by following the directions on the
    [instructions page](../tutorials/fundamentals/installation.md#install-as-python-package-recommended).

    napari supports different Qt backends, and you can choose which one to install and use.

    For example, for PyQt5, the default, you would use the following:
    ```sh
    pip install -e ".[pyqt,dev]"  # (quotes only needed for zsh shell)
    ```

    If you want to use PySide2 instead, you would use:
    ```sh
    pip install -e ".[pyside,dev]"  # (quotes only needed for zsh shell)
    ```

    Finally, if you already have a Qt backend installed or want to use an experimental one like Qt6 use:
    ```sh
    pip install -e ".[dev]"  # (quotes only needed for zsh shell)
    ```

    Note that in this last case you will need to install your Qt backend separately.

5. We use [`pre-commit`](https://pre-commit.com) to format code with
   [`black`](https://github.com/psf/black) and lint with
   [`ruff`](https://github.com/charliermarsh/ruff) automatically prior to each commit.
   To minimize test errors when submitting pull requests, please install `pre-commit`
   in your environment as follows:

   ```sh
   pre-commit install
   ```

   Upon committing, your code will be formatted according to our [`black`
   configuration](https://github.com/napari/napari/blob/main/pyproject.toml), which includes the settings
   `skip-string-normalization = true` and `max-line-length = 79`. To learn more,
   see [`black`'s documentation](https://black.readthedocs.io/en/stable/).

   Code will also be linted to enforce the stylistic and logistical rules specified
   in our [`flake8` configuration](https://github.com/napari/napari/blob/main/setup.cfg), which currently ignores
   [E203](https://lintlyci.github.io/Flake8Rules/rules/E203.html),
   [E501](https://lintlyci.github.io/Flake8Rules/rules/E501.html),
   [W503](https://lintlyci.github.io/Flake8Rules/rules/W503.html) and
   [C901](https://lintlyci.github.io/Flake8Rules/rules/C901.html).  For information
   on any specific flake8 error code, see the [Flake8
   Rules](https://lintlyci.github.io/Flake8Rules/).  You may also wish to refer to
   the [PEP 8 style guide](https://peps.python.org/pep-0008/).

   If you wish to tell the linter to ignore a specific line use the `# noqa`
   comment along with the specific error code (e.g. `import sys  # noqa: E402`) but
   please do not ignore errors lightly.

Now you are all set to start developing with napari.

## Contributing documentation

If you wish to contribute documentation changes to napari, please read the [guide on contributing documentation](documentation/index.md). 

## Adding icons

If you want to add a new icon to the app, make the icon in whatever program you
like and add it to `napari/resources/icons/`.  Icons must be in `.svg` format.

Icons are automatically built into a Qt resource file that is imported when
napari is run.  If you have changed the icons and would like to force a rebuild
of the resources, then you can either delete the autogenerated
`napari/resources/_qt_resources*.py` file, or you can set the
`NAPARI_REBUILD_RESOURCES` environmental variable to a truthy value, for
example:

```sh
export NAPARI_REBUILD_RESOURCES=1
```

Icons are typically used inside of one of our `stylesheet.qss` files, with the
`{{ id }}` variable used to expand the current theme name.

```css
QtDeleteButton {
   image: url("theme_{{ id }}:/delete.svg");
}
```

### Creating and testing themes

A theme is a set of colors used throughout napari.  See, for example, the
builtin themes in `napari/utils/theme.py`.  To make a new theme, create a new
`dict` with the same keys as one of the existing themes, and
replace the values with your new colors.  For example

```python
from napari.utils.theme import get_theme, register_theme


blue_theme = get_theme('dark')
blue_theme.update(
    background='rgb(28, 31, 48)',
    foreground='rgb(45, 52, 71)',
    primary='rgb(80, 88, 108)',
    current='rgb(184, 112, 0)',
)

register_theme('blue', blue_theme)
```


To test out the theme, use the
`qt_theme_sample.py` file from the command line as follows:

```sh
python -m napari._qt.widgets.qt_theme_sample
```
*note*: you may specify a theme with one additional argument on the command line:
```sh
python -m napari._qt.widgets.qt_theme_sample dark
```
(providing no arguments will show all themes in `theme.py`)

## Translations

Starting with version 0.4.7, napari codebase include internationalization
(i18n) and now offers the possibility of installing language packs, which
provide localization (l10n) enabling the user interface to be displayed in
different languages.

To learn more about the current languages that are in the process of
translation, visit the [language packs repository](https://github.com/napari/napari-language-packs)

To make your code translatable (localizable), please use the `trans` helper
provided by the napari utilities.

```python
from napari.utils.translations import trans

some_string = trans._("Localizable string")
```

To learn more, please see the {ref}`translations guide <translations>`.

## Making changes

Create a new feature branch:
```sh
git checkout main -b your-branch-name
```

`git` will automatically detect changes to a repository.
You can view them with:
```sh
git status
```

Add and commit your changed files:
```sh
git add my-file-or-directory
git commit -m "my message"
```

## Tests

We use unit tests, integration tests, and functional tests to ensure that
napari works as intended. Writing tests for new code is a critical part of
keeping napari maintainable as it grows.

We have dedicated documentation on [testing](napari-testing) that we recommend you
read as you're working on your first contribution.

### Help us make sure it's you

Each commit you make must have a [GitHub-registered email](https://github.com/settings/emails)
as the `author`. You can read more [here](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address).

To set it, use `git config --global user.email your-address@example.com`.

## Keeping your branches up-to-date

Switch to the `main` branch:
```sh
git checkout main
```

Fetch changes and update `main`:
```sh
git pull upstream main --tags
```

This is shorthand for:
```sh
git fetch upstream main --tags
git merge upstream/main
```

Update your other branches:
```sh
git checkout your-branch-name
git merge main
```

## Sharing your changes

Update your remote branch:
```sh
git push -u origin your-branch-name
```

You can then make a
[pull-request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request) to `napari`'s `main` branch.

## Code of Conduct

`napari` has a [Code of Conduct](napari-coc) that should be honored by everyone who participates in the `napari` community.

## Questions, comments, and feedback

If you have questions, comments, suggestions for improvement, or any other inquiries
regarding the project, feel free to open an [issue](https://github.com/napari/napari/issues).

Issues and pull-requests are written in [Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/about-writing-and-formatting-on-github).
You can find a comprehensive guide [here](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).
