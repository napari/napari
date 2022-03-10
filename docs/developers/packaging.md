(napari-packaging)=

# Packaging

Once a release is cut, napari is distributed in two main ways:

* Packages: both to PyPI and conda-forge.
* Installers: bundles that include napari plus its runtime dependencies in a step-by-step
  executable.

## Packages

Despite its numerous dependencies, `napari` itself is a simple Python project that can be packaged
in a straight-forward way.

Creating and submitting the packages to PyPI (the repository you query when you do `pip install`)
is handled in the [`.github/workflows/make_release.yml`][2] workflow. Creation is
handled with `make dist` (as specified in our [`Makefile`][3]) and submission is done using the
[official PyPA GitHub Action][4]. This workflow will also create a GitHub Release.

Once the Python package makes it to PyPI, it will be picked by the `conda-forge` bots, which
will automatically submit a PR to the [`napari-feedstock`][1] repository. This is all automated
by the `conda-forge` infrastructure (see [previous examples][16]), so we only need to check that
the metadata in the recipe has been adjusted for the new release. Pay special attention to the
runtime dependencies and version strings!

Once the conda-forge CI is passing and the PR is approved and merged, the final packages will be
built on the default branch and uploaded to the `conda-forge` channel. Due to the staging steps and
CDN synchronization delays, the conda packages can take up to 2h to be available.

### Nightly packages

We also build nightly packages off `main` and publish them to the `napari/label/nightly` channel.
These are the same packages that are used in the `constructor` installers (see below), so their CI
is specified in `.github/workflows/bundle_conda.py`.

To do it in a `conda-forge` compatible way, we actually _clone_ `napari-feedstock` and patch the
source instructions so the code is retrieved from the repository branch directly. The version is
also patched to match the `setuptools-scm` string. After [rerendering][8] the feedstock, we run
`conda-build` in the same way `conda-forge` would do and upload the resulting tarballs to our
[Anaconda.org channel][17].

Additionally, the tarballs are also passed as artifacts to the next stage in the pipeline: building
the `constructor` installers (more below).

## Installers

Once the packages have been built and uploaded to their corresponding repositories, we can bundle
them along with their dependencies in a single executable that end users can run to install napari
on their systems, with no prior knowledge of `pip`, `conda`, virtual environments or anything.

A software installer is usually expected to fulfill these requirements:

* It will install the application so it can be run immediately after.
* It will provide a convenient way of opening the application, like a shortcut or a menu entry.
* It will allow the user to uninstall the application, leaving no artifacts behind.

Right now, we are using two ways of generating the installers:

* With `briefcase`, which takes PyPI packages.
* With `constructor`, which takes `conda` packages.

`conda` packages offer several advantages when it comes to bundling dependencies, since it makes
very little assumptions about the underlying system installation. As a result, `constructor` bundles
include libraries that might be missing in the target system and hence should provide a more robust
user experience.

### Briefcase-based installers

[`briefcase`][5] based installers are marked for deprecation so we will not discuss them here.
If you are curious, you can check `bundle.py` and `.github/workflows/make_bundle.yml` for
details.

### Constructor-based installers

[`constructor`][6] allows you to build cross-platform installers out of `conda` packages. It
supports the following installer types:

* On Linux, a shell-based installer is generated. Users can execute it with `bash installer.sh`.
* On macOS, you can generate both PKG and shell-based installers. PKG files are graphical installers
  native to macOS, and that's the method we use with napari.
* On Windows, a graphical installer based on NSIS is generated.

The configuration is done through a `construct.yaml` file, documented [here][7]. We generate one on
the fly in the `bundle_conda.py` script. Roughly, we will build this configuration file:


```yaml
# os-agnostic configuration
name: napari
version: 0.4.12  # depending on workflow triggers, this can also be a dev snapshot for nightlies
company: Napari
license: <points to a generated license that bundles the licenses of all dependencies>
channels:
  # - local  # only in certain situations, like nightly installers where we build napari locally
  - napari/label/bundle_tools  # temporary location of our forks of the constructor stack
  - conda-forge
specs:
  - napari
  - napari-menu  # provides the shortcut configuration for napari
  - python 3.7  # pinned to the version of the running interpreter, configured in the CI
  - conda  # we add these to assist in the plugin installations
  - mamba  # we add these to assist in the plugin installations
  - pip    # we add these to assist in the plugin installations
menu_packages:
  - napari-menu  # don't create shortcuts for anything else in the environment

# linux-specific config
default_prefix: $HOME/napari-0.4.12  # default installation path

# macos-specific config
name: napari-0.4.12  # override this because it's the only way to encode the version in the default
                     # installation path
installer_type: pkg  # otherwise, defaults to sh (Linux-like)
welcome_image: resources/napari_1227x600.png  # bg image with the napari logo on bottom-left corner
welcome_file: resources/osx_pkg_welcome.rtf  # rendered text in the first screen
conclusion_text: ""  # set to an empty string to revert constructor customizations back to system's
readme_text: ""  # set to an empty string to revert constructor customizations back to system's
signing_identity_name: "Apple Developer ID: ..."  # Name of our installer signing certicate

# windows-specific config
welcome_image: resources/napari_164x314.png  # logo image for the first screen
header_image:  resources/napari_150x57.png  # logo image (top left) for the rest of the installer
icon_image: napari/resources/icon.ico  # favicon for the taskbar and title bar
default_prefix: %USERPROFILE%/napari-0.4.12  # default location for user installs
default_prefix_domain_user: %LOCALAPPDATA%/napari-0.4.12  # default location for network installs
default_prefix_all_users: %ALLUSERSPROFILE%/napari-0.4.12  # default location for admin installs
signing_certificate: certificate.pfx  # path to signing certificate
```

On the OS-agnostic keys, the main keys are:

* `channels`: where the packages will be downloaded from. We mainly rely on conda-forge for this,
  where `napari` is published. However, we also have `napari/label/bundle_tools`, where we store
  our `constructor` stack forks (more on this later). In nightly installers, we locally build our
  own development packages for `conda` without resorting to `conda-forge`. To make use of those
  (which are eventually published to `napari/label/nightly`), we unpack the GitHub Actions artifact
  in a specific location that `constructor` recognizes as a _local_ channel once indexed.
* `specs`: the conda packages that should be provided by the installer. Constructor will perform a
  conda solve here to retrieve the needed dependencies.
* `menu_packages`: restrict which packages can create shortcuts. We only want the shortcuts provided
  by `napari-menu`, and not any that could come from the (many) dependencies of napari.

Then, depending on the operating systems and the installer format, we customize the configuration
a bit more.

#### Default installation path

This depends on each OS. Our general strategy is to put it under `napari-<version>` in the user
directory. However, there are several constrains we need to take into account to make this happen:

* On Windows, users can choose between an "Only me" and "All users" installation. This changes what
  we understand by "user directory". This is further complicated by the existence of "domain users",
  which are not guaranteed to have a user directory per se.
* On macOS, the PKG installer does not offer a lot of flexibility for this configuration, so we
  need to override the _name_ to include the version tag.

#### Branding

Graphical installers can be customized with logos and icons. These images are stored under the
`resources/` directory (outside of the source), with the exception of the square logos/icons (which
are stored under `napari/resources/` so the shortcuts can find them after the installation).

Some of the steps are also configured to display a custom text, like the license or the welcome
screen on macOS.

> Note: Most of the branding for macOS is only available in our fork. This has been sent upstream
> but it's not accepted or released yet.

#### Signing

In order to avoid security warnings on the target platform, we need to sign the generated installer.

On macOS, once Apple's _Installer Certificate_ has been installed to a keychain and unlocked
for its use, you can have `constructor` handle the signing via `productsign` automatically.
However, this is not enough for a warning-free installation, since its contents need to be
_notarized_ and _stapled_ too. For this reason, `constructor` has been modified to also
`codesign` the bundled `_conda.exe` (the binary provided by conda-standalone, see below) with
the _Application Certificate_. Otherwise, notarization fails. After that, two actions take care
of notarizing and stapling the resulting PKG.

On Windows, any Microsoft-blessed certificate will do. Our `constructor` fork allows us to specify
a path to a PFX certificate and then have the Windows SDK `signtool` add the signature. Note that
`signtool` is not installed by default on Windows (but it is on GitHub Actions).


#### Details of our `constructor` stack fork

Many of the features here listed were not available on `constructor` when we started working on it.
We have added them to the relevant parts of the stack as needed, but that has resulted in a lot of
moving pieces being juggled to make this work. Let's begin by enumerating the stack components:

1. `constructor` is the command-line tool that _builds_ the installer. It depends on `conda` to
   solve the `specs` request. It also requires a copy of `conda-standalone` (a PyInstaller-frozen
   version of `conda`) to be present at build time so it can be bundled in the installer. This
   is needed because that `conda-standalone` copy will handle the extraction, linking and
   shortcut creation when the user runs the installer on their machine.
2. `conda-standalone` is a frozen version of `conda`. Among its dependencies, we can find
   `menuinst`, which handles the creation of shortcuts and menu entries.
4. `menuinst` was only used on Windows before our work, so we basically rewrote it to handle
   cross-platform shortcuts.
3. `conda` interfaces with `menuinst` to delegate the shortcut creation. Since this was only enabled
   on Windows, we needed to unlock the other platforms and rewrite the parts that assumed Windows
   only behaviour. Surprise, this involved custom solver behaviour too!

Since `menuinst` is frozen together with `conda` for `conda-standalone`, every little change in any
of those requires a rebuild of `conda-standalone` so `constructor` can find the new version during
the installer creation. As a result, we needed to fork _and repackage_ all four components!

Notice the repackaging needs. It's not enough to fork and patch the code. We also need to create
the conda packages and put them in a channel so the downstream dependencies can pick them when they
are rebuilt. This repackaging is done through a separate `conda-forge` clone that only handles our
forks. It is configured to use GitHub Actions (instead of Azure) and upload to the `napari` channel
(instead of `conda-forge`).

For example, if a patch is introduced in `menuinst`, the following needs to happen before it makes
it to the final napari installer:

1. Write and test the patch. Make sure it passes its own CI.
2. Make sure `conda` still works with the new changes. It needs to call `menuinst` after all.
3. Create the `menuinst` package and upload it to Anaconda.org.
4. Rebuild and upload `conda-standalone` so it picks the new `menuinst` version.
5. Trigger the napari CI to build the new installer.

Very fun! So where do all these packages live?

| Package            | Fork                                            | Feedstock                                          |
|--------------------|-------------------------------------------------|----------------------------------------------------|
| `constructor`      | [jaimergp/constructor @ `menuinst+branding`][9] | [jaimergp-forge/constructor-feedstock][12]         |
| `conda-standalone` | _N/A_                                           | [conda-forge/conda-standalone-feedstock PR#21][13] |
| `conda`            | [jaimergp/conda @ `cep-menuinst`][10]           | [jaimergp-forge/conda-feedstock][14]               |
| `menuinst`         | [jaimergp/menuinst @ `cep`][11]                 | [jaimergp-forge/menuinst-feedstock][15]            |


Most of the forks live in `jaimergp`'s account, under a non-default branch. They are published
through the `jaimergp-forge` every time a commit to `master` is made. Versions are arbitrary here,
but they are set to be greater than the latest official version, and the `build` number is
incremented for every rebuild.

The only exception is `conda-standalone`. It doesn't have its own repository or fork because it's
basically a repackaged `conda` with some patches. Those patches live in the feedstock only. The
other difference is that the feedstock does not live in `jaimergp-forge`, but just as draft PR in
the `conda-forge` original feedstock. This is because, for some reason, if `conda-standalone` is
built on GitHub Actions machines, the Windows version will fail with `_ssl` errors which do not
appear in Azure. For this reason, the CI is run as normal on `conda-forge`, and then the artifacts
are retrieved from the Azure UI and manually uploaded to the `napari` channel. Fun again!

_Eventually_ all these complexities will be gone because all of our changes will have been merged
upstream. For now, this not the case. Speaking of which, what are our changes? Below you can find
a high-level list of the main changes introduced in the stack.

##### Changes in `menuinst`

* Add cross-platform specification for shortcut configuration
* Enable support on Windows, Linux and macOS
* Re-engineer environment activation
* Maintain backwards compatibility with Windows
* Simplify API
* Remove CLI

##### Changes in `conda`

* Add API support for menuinst v2
* Enable code paths for non-Windows Platforms
* Fix shortcut removal logic
* Add `--shortcuts-only` flag to support `menu_packages` constructor key natively

##### Changes in `conda-standalone`

* Unvendor menuinst patches
* Do not vendor constructor NSIS scripts
* Adapt `conda constructor` entry point for the new menuinst API

##### Changes in `constructor`

* Use `--shortcuts-only`
* Add branding options for macOS PKG installers
* Always leave `_conda.exe` in the install location
* Do not offer options for conda init or PATH manipulations (these should be Anaconda specific)
* Add signing for Windows
* Add notarization for macOS PKG

<!-- hyperlinks -->

[1]: https://github.com/conda-forge/napari-feedstock
[2]: https://github.com/napari/napari/blob/main/.github/workflows/make_release.yml
[3]: https://github.com/napari/napari/blob/main/Makefile#L20
[4]: https://github.com/pypa/gh-action-pypi-publish
[5]: https://github.com/beeware/briefcase
[6]: https://github.com/conda/constructor
[7]: https://github.com/conda/constructor/blob/master/CONSTRUCT.md
[8]: https://conda-forge.org/docs/maintainer/updating_pkgs.html#rerendering-feedstocks
[9]: https://github.com/jaimergp/constructor/tree/menuinst+branding
[10]: https://github.com/jaimergp/conda/tree/cep-menuinst
[11]: https://github.com/jaimergp/menuinst/tree/cep
[12]: https://github.com/jaimergp-forge/constructor-feedstock
[13]: https://github.com/conda-forge/conda-standalone-feedstock
[14]: https://github.com/jaimergp-forge/conda-feedstock
[15]: https://github.com/jaimergp-forge/menuinst-feedstock
[16]: https://github.com/conda-forge/napari-feedstock/pulls?q=is%3Apr+sort%3Aupdated-desc+is%3Aclosed
[17]: https://anaconda.org/napari