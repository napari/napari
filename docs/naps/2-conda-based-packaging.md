(nap2)=

# NAP-2 — Distributing napari with conda-based packaging

```{eval-rst}
:Author: Jaime Rodríguez-Guerra, Gonzalo Peña-Castellanos
:Created: 2022-05-05
:Resolution: https://github.com/napari/napari/pull/4602#issuecomment-1184061072
:Resolved: 2022-07-14
:Status: Draft
:Type: Standards Track
:Version effective: N/A
```

## Abstract

napari can be installed through different means, including `pip` and `conda`. However, users
not exposed to package managers and command-line interfaces might prefer other options more
native to their operating system of choice. This is usually achieved through graphical
installers with a step-by-step interface.

This NAP discusses how we will use and adapt tools borrowed from the `conda` packaging
world to build platform-specific installers and implement update strategies for napari and
its plugin ecosystem.

## Motivation and Scope

napari is packaged for PyPI [^pypi-napari] and conda-forge [^napari-feedstock] and can be
installed with any client supporting these repositories (e.g., `pip` and `conda`,
respectively). For a number of releases, platform-specific installers were also provided with
Briefcase [^briefcase].

Briefcase relies on PyPI packaging to assemble the installer. The PyPI packaging strategy,
however, presents a series of limitations for the napari ecosystem:

* No standardized building infrastructure. PyPI accepts submissions from any user without requiring
  any validation or review. As a result, packages can be built using arbitrary toolchains or
  expecting different libraries in the system. `cibuildwheel` [^cibuildwheel] and related tools
  [^audithwheel] [^delocate] [^delvewheel] can definitely help users who want
  to follow community packaging practices,
  but there is no guarantee of it being used by individual packages.
  This can result in ABI incompatibilities with
  the target system and within the plugin ecosystem, especially when some packages vendor specific
  libraries [^pypi-parallelism-abi].
* PyPI metadata is often not detailed enough. This is a byproduct of the previous point, which
  makes it difficult for the different clients (pip, poetry, etc) to guarantee that the
  resulting installation is self-consistent and all the packages involved are compatible with
  each other.
* PyPI is Python specific. While it's possible to find non-Python packages in the repository,
  it was not designed to do so. In scientific projects, researchers often combine packages from
  different languages to build their pipeline. If we want a thriving plugin ecosystem,
  restricting the packaging options to a language-specific repository can be limiting.
* PyPI only provides Python _packages_. It does not distribute Python itself, leaving that to the
  installer infrastructure. In the case of Briefcase, this is obtained via their own distribution
  mechanisms [^briefcase-python]. This presents one more moving piece that can
  result in incompatibilities with the target system if not controlled properly
  (see issues [^appimage-crash][^appimage-crash2]).

In contrast, `conda`-based packaging offers some benefits in those points:

* `conda` is language agnostic. It can package things for Python and other languages, but also the
  language runtimes/interpreters themselves! This allows plugins to depend on compiled
  libraries with no Python counterparts or wrappers, or maybe even different language interpreters.
* `conda` maintains its own package metadata separate from PyPI, allowing solvers to do their
  work better and more efficiently. It can also be patched after a package is released,
  allowing corrections to be made over time without building new artifacts.
* `conda-forge` is a community-driven effort to build `conda` packages in a supervised and
  automated way that ensures binary compatibility across packages and languages. Every
  submission needs to be reviewed and approved by humans after successfully passing the CI.
  This adds guarantees for provenance, transparency and debugging.
* `conda` has the notion of optional version constrains. A package can provide constraints for other
  packages that _could_ be installed alongside, without depending on them. This offers a lot of
  flexibility to manage a plugin ecosystem with potentially wildly different requirements, which
  would risk conflicts.

`conda` packaging also has its own downsides compared to `pip`, though:

* pip and PyPI are the *de facto* standard in Python packaging, which means
  more developers are aware of them and familiar with them, and how to create
  packages for them. Conda in contrast presents a community education
  challenge.
* Even with a well-developed community of practice among napari and napari
  plugin developers, some significant industry-provided packages, such as
  Apple's tensorflow-metal, may *never* be available no conda-forge.
* Although the conda-forge review process is an advantage with regards to
  correctness and reliability, it presents a scalability challenge in the
  absence of broader community education about conda packaging.
* pip can install packages from a simple local directory, from a zip file, or
  from a GitHub repository. These simple installation methods are favored by
  small labs and institutions that want to create plugins for internal use,
  rather than for broad distribution.

This NAP proposes to add a `conda`-based distribution mechanism for the napari
application and plugins, supported by five key milestones:

1. Distributing napari and plugins on conda-forge
2. Building conda-based installers for napari
3. Adding support for conda packages in the plugin manager
4. Enabling in-app napari version updates
5. Deprecating Briefcase-based installers

Throughout the process, we will aim to minimize conda's downsides by providing
local conda-based installation options and documentation about how to use them.
We will also provide users an opt-in, "use at your own risk" method to install
PyPI packages where they do not exist on conda-forge.

## Detailed Description

The details for each milestone will be discussed in subsections.

### Milestone 1: Distributing napari and plugins on conda-forge

napari 0.2.12 was submitted to conda-forge [^staged-recipes-napari] in Oct 2019 and the PR was
merged some months later. As a result, napari is available on conda-forge since Feb 2020
[^napari-feedstock-creation]. The conda-forge bots auto-submit PRs to build the new versions
once detected on PyPI. This means that releases on conda-forge can slightly lag behind PyPI.
To avoid accidental delays in the releases, conda-forge packaging needs to be considered part
of the release guide [^release-guide].

Pre-release packages are additionally built in our CI by cloning the conda-forge feedstock and
patching it to use the local source. The artifacts are uploaded to the `napari` channel at
Anaconda.org [^napari-channel].

While napari itself has been on conda-forge for some years now, until recently, the plugin
ecosystem still broadly relied on PyPI. In the case of napari users that relied on conda packages,
that means that the plugin manager would use `pip` to install the plugin and its dependencies in
the conda environment, potentially mixing PyPI packages with conda-forge packages and causing
conflicts due to binary incompatibilities.

To avoid this risk, this NAP recommends packaging all existing napari plugins (and
their dependencies) on conda-forge too. This (ongoing) effort started in Jan 2022, resulting
in ~200 pull requests (PRs) to date [^staged-recipes-all-plugins].

That said, that is only the initial migration. We need a way to ensure that new plugins are also
packaged on conda-forge. We recommend adding it to the plugin development documentation, as well
as adding support for the relevant metadata on napari hub.

Lastly, in order to ensure maximum compatibility across plugins, the napari project should also
provide documentation and guidelines on what versions of major scientific packages are supported
on each napari release. For example, we should control the version bounds for `numpy`,
`scikit-image` and similar members of the PyData ecosystem. Otherwise, we might arrive to a
situation where plugin developers are choosing wildly different `numpy` versions for their projects,
making them non-installable together. In conda jargon, the set of conditional restraints are called
pinnings and implemented as part of a metapackage (a package that doesn't distribute files, only
provides metadata). From now on we will refer to them as _napari pinnings_.

> There are precedents in other projects that support the usage of pinnings. For example,
> Fiji/SciJava [^scijava-pinnings], conda-forge [^conda-forge-pinnings], or Maxiconda [^maxiconda].

A prototype notebook assessing the "installability" of all plugins in the same environment
is available [here](https://colab.research.google.com/drive/1QxbBZYe9-AThGuRsTfwYzT72_UkamXmk)
Results across Python versions show incompatibilities on Linux, possibly even more intricate on macOS
and Windows. Excerpt for Python 3.9:

```
$ mamba create -n test-napari-installability --dry-run -q napari=0.4.15 python=3.9 affinder \
  bbii-decon brainglobe-napari-io [...] nfinder platelet-unet-watershed smo waver workshop-demo
Encountered problems while solving.
Problem: nothing provides __linux needed by dask-cuda-21.10.0-pyhd8ed1ab_0
Problem: package napari-subboxer-0.0.1-pyhd8ed1ab_0 requires napari 0.4.12, but none of the providers can be installed
Problem: package napari-tomoslice-0.0.7-pyhd8ed1ab_0 requires napari 0.4.12, but none of the providers can be installed
Problem: package napari-nikon-nd2-0.1.3-pyhd8ed1ab_0 requires python >=3.6,<=3.9, but none of the providers can be installed
Problem: package napari-multitask-0.0.2-pyhd8ed1ab_0 requires python >=3.8, but none of the providers can be installed
Problem: nothing provides __cuda needed by tensorflow-2.7.0-cuda102py310hcf4adbc_0
Problem: package napari-console-0.0.4-pyhd8ed1ab_0 requires ipykernel >=5.2.0, but none of the providers can be installed
```

#### Potential Risks

Lack of involvement of the community might mean that only one source of packaging is updated often,
forcing napari core developers to take over in the maintenance of the conda packages.

Notably, best practices for PyPI packaging are sometimes ignored by some plugins, which means
that they are not suitable for conda-forge packaging, which has stricter standards for
publication. The list of problems include vendoring other packages, licensing issues, binary
redistribution, inadequate dependencies metadata (too strict or too vague). The packaging team
at napari can help here, but this will not scale if the plugin ecosystem keeps growing.

As a result, some plugins might end up being available on PyPI but not on conda-forge. This further
reinforces the idea that conda-forge packaging is a second-class citizen for the plugin ecosystem.
This NAP recommends including packaging guidelines as part of the documentation for plugin
developers to alleviate these issues. That said, PyPI packages will still be allowed as a fallback
option for those projects that are not (yet) available on conda-forge.

### Milestone 2: Building conda-based installers for napari

Anaconda releases their Anaconda and Miniconda distributions with platform specific installers:

* On Windows, they offer an EXE built with NSIS
* On Linux, a text-based installer is offered as a fat SH script
* On macOS, a native, graphical PKG installer is provided in addition to the text-based option

These three products are created using `constructor` [^constructor], their own tool to gather
the required conda dependencies and add the logic to install them on the target machine.
However, `constructor` hasn't been well maintained in recent years (only small fixes),
which means that some work is needed to meet our needs. More specifically:

* Application shortcut creation is only supported on Windows
* PKG installers are created with hardcoded Anaconda branding
* Some conda-specific options cannot be removed (only disabled by default), which might distract
  users in the installers

In order to have `constructor` cover our needs, we need to add the features ourselves. Upstream
maintenance is expected to improve in the coming years, but for now the reviews are coming in slow. As
a result, we are temporarily forking the project and developing the features as needed while
submitting PRs upstream [^constructor-upstream] to keep things tidy. Our improved `constructor`
fork has the following features:

* Cross-platform shortcut creation for the distributed application thanks to a complete `menuinst`
  [^menuinst] rewrite
* PKG installers on macOS can be customized, signed and notarized
* `constructor` can deploy more than one conda environment at once now
* Extra files can be added to the installation without having to use an extra conda package
* Improvements for license collection and analysis
* Small fixes for reported installation size, progress reports and other cosmetic details

The resulting product now is able to install napari across operating systems with fully working
shortcuts that respect the activation mechanism of `conda` environments to guarantee that all
dependencies work properly. Having `constructor` distribute multiple environments at once allow us
to split the installation like this:

* A `base` environment with `conda`, `mamba` and `pip` to manage the other `conda` environments.
* A `napari-X.Y.Z` environment (X.Y.Z being the version of the bundled release) with napari and
  its dependencies.

This separation allows us to:

* Handle napari updates by simply creating a fresh environment with the new version (more on this
  in Milestone 4).
* Apply reparations in the environment without worrying that a faulty plugin install can render
  the whole environment non-functional.
* If necessary, adding support for "napari projects": a napari installation with a specific
  combination of plugins. Useful in the event that a user wants to use plugins that cannot be
  installed in the same environment due to conflicting dependencies (more details on this in
  Milestone 3).

The installer relies on conda-forge to obtain the needed packages. Pre-release installers
can be built thanks to the nightlies available on the napari channel in Anaconda.org.

### Milestone 3: Adding support for conda packages in the plugin manager

napari has its own plugin manager, which so far has relied on `pip` to install packages available on
PyPI. To make it compatible with conda packaging, three key changes are needed:

1.  The list of packages on conda-forge does not necessarily match the one coming from PyPI. Right
    now, the plugins on conda-forge are a _subset_ of those on PyPI, but this might change in the
    future if some napari plugins become available on conda-forge but not PyPI due to packaging
    limitations (e.g. availability of dependencies). As a result, the plugin manager needs to source
    the list of plugins from a repository-agnostic source: the napari hub API. It must be noted that
    napari hub currently uses PyPI as the ground truth for the list of published plugins and the
    available versions.
2.  Once the napari hub API is feeding the list, the plugin manager will also list those available
    on conda-forge. Packages that are only available on PyPI can also be installed as long as:
      * The dependencies of the PyPI package are on conda-forge
      * The PyPI package is pure Python (no compiled libraries)
    In the future, we might explore how to deal with PyPI packages within conda in a safer way, but
    this is an open packaging question that is extremely difficult to tackle robustly.
3.  Instrument the plugin manager backend so it can use `conda` or `mamba` to run the plugin
    installation, update or removal. Some level of customizability is needed to configure extra
    channels (e.g. a laboratory published their conda packages into their own private channel) and
    local sources (e.g. drag&drop a conda tarball).
4.  Add some control to the dependency landscape of the plugin ecosystem using the
    `napari-pinnings` metapackage mentioned in Milestone 1.

There are some technical limitations we need to work out as well, namely:

* Some plugin updates might fail because some files are in use already. For example, a plugin
  requires a more recent build of numpy (still allowed in our pinnings); however, numpy has been
  imported already and Windows has blocked the library files. An off-process update will be needed
  on Windows for the installation to succeed. On Unix systems this might not be a problem, but the
  update will still be incomplete without a napari restart (because `numpy` was already imported).
  This can be solved by watching the imported modules and the files involved in the conda
  transaction. On Windows, we can write a one-off activation script that will run before `napari`
  starts the next time. On Unix systems, a notification saying "Restart needed for full effect"
  might be enough.
* The plugin manager was designed to install one package at a time with `pip`. We have extended it
  to use `conda` or `mamba`, but it still works on a package-by-package basis. It would be preferred
  to offer the possibility of installing several packages together for more efficient solves, but
  this involves some UI/UX research first.

### Milestone 4: Enabling in-app napari version updates

The installers produced by `constructor` are designed to avoid existing installations.
Updating a `conda` environment involves more actions than just overwriting some files. For example,
some packages might feature uninstallation scripts that wouldn't be executed or cleaned up. For
this reason, Anaconda users are recommended to run `conda` itself to handle the update.

Our preferred approach is to create a fresh environment for the new version of napari. The reasons
are multiple:

* Performance and solving complexity: `conda` keeps track of the actions performed in the
  environment in a `conda-meta/history` file. The contents of this file are parsed by the solver,
  which prioritizes the packaged listed there in the solves. For example: if you started your
  environment with `conda create -n something python=3.9 numpy=1.21`, `numpy` will be _soft-pinned_
  during the lifetime of the environment. Soft-pinning means that the solver will try to respect
  that specific version (1.21) in every solve, unless the user asks for a different version
  explicitly (thus overriding the historic preference). For napari, this means that every plugin
  installation will be recorded in the history file, accumulating over time. If we compound this
  with `napari` updates, the problem gets larger with every new release.
* Guarantee of success: updating an environment to the latest napari release might not work right
  away, specially if the user has installed plugins that have conflicting packaging metadata. Even
  if the installation succeeds, insufficient/incorrect metadata might result in the wrong packages
  being installed, rendering the napari installation non-operational!
* Using several versions side-by-side: the multiple environments setup fits nicely in contexts
  where several napari versions (or even "flavors") are needed.

Since we install plugins to the napari environments, we need to guarantee that the new version
will be compatible with the installed plugins. To do so, we instruct `conda` to perform a _dry-run_
install. If the environment can be solved, we can proceed with no interruptions. If the environment
is not solvable, then we can offer the user two options:

* Start a fresh environment only containing napari, with no extra plugins, keeping the old version
  in place.
* (Experimental idea / suggestion) Run a potentially time-consuming analysis on which package(s)
  are creating the installation conflicts and suggest which plugins cannot be installed. This kind
  of analysis would entail a series of dry runs, removing one plugin at a time, hoping that the
  environment gets solved eventually.

It must be considered that better metadata at napari hub could substantially improve the
installability analysis for the end-user. If we keep track of which napari versions the plugins are
compatible with (either by running some kind of CI ourselves or making this analysis part of the
submission procedure), we could simply query the API to anticipate which packages are installable
before running the update.

#### Potential risks

Co-installability of plugins is ultimately a matter of metadata and good practices. The risks here
are similar to the concerns shared in Milestone 1.

### Milestone 5: Deprecating Briefcase-based installers

Once we are satisfied with how `conda` installers are working for the end user, we can think of
removing the pipelines that build the Briefcase-based installers. Ideally, we mark them for
deprecation on one release, and then completely remove them on the next one. This involves
deleting the build scripts, the CI workflows, some metadata in `setup.cfg` and possibly some
utility functions [^briefcase-utilities-example] and code paths [^briefcase-workaround-example]
in napari itself.

## Related Work

napari has been using Briefcase to build its installers so far, as discussed in the "Motivation and
scope" section. Other alternatives we considered before choosing `conda` were:

* Adapting Briefcase to use `conda` packages, but it felt like we were rewriting `constructor` at
  that point, only to get a barely working prototype.
* Freezing `napari` directly with PyOxidizer [^pyoxidizer] and Nuitka [^nuitka]. These experiments
  were not successful either and didn't allow for a good plugin installation story (the frozen
  executables are immutable).

## Implementation

This NAP has described the whole strategy to implement a successful and comprehensive conda
packaging story for napari. This work involves many moving pieces across different projects and
tools, hence why a single PR is out of the question. In the following sections, we will list the
relevant PRs opened so far. Before that, though, we will propose a general strategy on how this
infrastructure will be maintained and governed.

### General strategy

After many months operating on the `napari/napari` repository, we would like to propose the creation
of a new repository (e.g. `napari/packaging`), where all these efforts can take place without
getting in the way of the napari development itself.

Most packaging tasks do not (and should not) rely tightly on the software being packaged,
provided it follows the best practices. As such, the existing pipelines are mostly independent
and could already live outside of `napari/napari`. Actually, it could be argued that they
_should_ live outside to guarantee that the pipelines are not too tightly coupled.

Given that Milestone 4 could benefit from having a specific tool to handle updates and manage
existing installations (more details below), this repository could initially host its initial
prototype, which would be designed in a napari-agnostic way for easy reusability in other
projects facing our `constructor` improvements already [^mne-constructor].

### Milestone 1: Distributing napari and plugins on conda-forge

This work can be divided in two different tasks: adjusting the conda-forge feedstock for napari,
and then migrating all the plugins over to conda-forge.

* [Adjust outputs for multiple qt backends and menuinst=2](https://github.com/conda-forge/napari-feedstock/pull/32)
* [Plugin migration search query](https://github.com/conda-forge/staged-recipes/pulls?q=is%3Apr+author%3Agoanpeca+created%3A2022-01-01..2022-05-01)
  (~200 PRs)

At the time of writing, the plugin migration to conda-forge can be considered 90% done, but a long
tail of non-trivial packages need to be worked on. This is caused by vendored binaries,
non-compliant licensing schemes or bad packaging practices.

#### Tasks

* [ ] Add documentation about conda-forge in the release guide
* [ ] Add documentation about conda-forge in the plugin developer guide
* [ ] Ensure the cookiecutter template has some notion of conda packaging
* [ ] Decide which packages need to be governed by the _napari pinnings_ metapackage

### Milestone 2: Building conda-based installers for napari

Implementing the `constructor` workflow on napari was mainly done in a single, long-lived PR, that
has seen a couple of minor updates in the recent releases:

* Main PR: [#3378](https://github.com/napari/napari/pull/3378), superseded by
  [#3555](https://github.com/napari/napari/pull/3555) (Prototype a conda-based bundle)
* Other PRs:
    * [#3462](https://github.com/napari/napari/pull/3462) (Move icons to package source)
    * [#4185](https://github.com/napari/napari/pull/4185) (Add licensing page)
    * [#4210](https://github.com/napari/napari/pull/4210) (Fix EULA/licensing/signing issues)
    * [#4221](https://github.com/napari/napari/pull/4221) (Adjust conditions that trigger signing)
    * [#4307](https://github.com/napari/napari/pull/4307) (Test installers in CI)
    * [#4309](https://github.com/napari/napari/pull/4309) (Use conda-forge/napari-feedstock)
    * [#4387](https://github.com/napari/napari/pull/4387) (Fix unlink errors on cleanup)
    * [#4444](https://github.com/napari/napari/pull/4444) (Add versioning to installer itself)
    * [#4447](https://github.com/napari/napari/pull/4447) (Use custom `.condarc` file)
    * [#4525](https://github.com/napari/napari/pull/4525) (Revert to napari-versioned default paths)


Adding the missing pieces to `constructor` involves changes in four different projects:

* `conda/constructor` itself, the tool that builds the installer
* `conda/menuinst`, responsible of creating the shortcuts
* `conda/conda`, which relies on `menuinst` to delegate the shortcut creation, but this code path
  was only enabled on Windows so far
* `conda-forge/conda-standalone-feedstock`, which freezes a `conda` install along with their
  dependencies and adds a thin layer on top to communicate with the `constructor` installer during
  the installation process.

The full list of PRs is available on
[this issue comment](https://github.com/napari/napari/issues/4248#issuecomment-1066574137).
A complete list of features can be found in the packaging documentation [^napari-packaging-docs].

#### Tasks

* [ ] Create `napari/packaging` and migrate the conda bundle CI there, along with the relevant
      metadata
* [ ] Add detailed documentation about how the installers are structured and how they work
      internally
* [ ] Add PyPI-conda consistency checks to ensure the metadata at `napari/napari` and
      `napari/packaging` match

### Milestone 3: Adding support for conda packages in the plugin manager

Support for conda/mamba handling of plugin installations was implemented in a base PR and then
extended with different PRs:

* Initial PR: [#2943](https://github.com/napari/napari/pull/2943) (Add initial support to install
  plugins with conda/mamba)
* Other PRs that improved and extended the functionality:
    * [#3288](https://github.com/napari/napari/pull/3288) (Fix plugin updates)
    * [#3369](https://github.com/napari/napari/pull/3369) (Add cancel actions to plugin manager)
    * [#4074](https://github.com/napari/napari/pull/4074) (Use napari hub to list plugins)
    * [#4520](https://github.com/napari/napari/pull/4520) (Rework how subprocesses are launched)

Some more work is needed to offer full support to the plugin ecosystem, as detailed below.
#### Tasks

* [X] Make the `Installer` class  [^napari-installer-classs] conda/mamba-aware
* [X] Populate the plugin listing with data obtained from the napari hub API [^napari-hub-api]
* [ ] Detect which plugins can be installed directly from conda-forge and which ones need a
      combination of `conda` channels and PyPI sources
* [ ] Add support for custom conda channels and local sources in the UI
* [ ] Allow for simultaneous install of plugins. Currently multiple plugins can be selected for
      install, uninstall and update, but each on of these actions, are queued and run sequentially.

> Note: Using both `conda` and `pip` presents certain risks that could irreversibly disrupt the
> installation, so it needs to be studied very carefully. Dependencies should be
> provided from `conda-forge` whenever possible (see previous experiments on this kind of automation
> [^conda-pip-issue-comment] [^pamba]), but PyPI packages can still be used as long as the user
> consents to a warning detailing the risks and recovery options.

### Milestone 4: Enabling in-app napari version updates

Our first attempts [^in-app-update-pr] to implement this consisted of a pop-up dialog that would
notify the user about the available updates. Accepting the dialog would trigger the creation of a
new environment (e.g. `napari-0.4.16`) alongside the existing one (e.g. `napari-0.4.15`). While it
could be argued that this is not an "update" per se, but a new installation, it fulfills the same
goal: the new version is available. If the user wants to get rid of the old one, an option can
be offered to auto-clean on success, or after not using it for some configurable time.

However, having several versions of napari coexisting might risk questions like "which napari
version is the default one?", "which one will be in charge of updating napari?", "what if the
update logic changes over time?". There possible workarounds, but from the user perspective it could
get confusing very fast.

For that reason, we decided to slightly redesign our approach to in-app updates. We will still use
several environments to keep things tidy and performant (see "Detailed description" for Milestone 2
for more details), but the update checks will mimic something closer to a local server-client model
or, more accurately, a `napari-updater` process (name subject to change; other options include
`napari-launcher` or  `napari-manager`) that can be queried by a running `napari` instance.

* `napari-updater`: We will create a separate standalone package to handle updates outside of the
  `napari` process. This will be designed in an application agnostic way while targeting the needs
  of the napari project. It will allow the user to handle updates but also manage the existing
  napari installations (investigate issues, remove old ones, clean expired caches, etc).
* `napari` itself: Each napari version will have a very basic notion about `napari-updater` and
  will query for updates every time it starts, and also periodically while running. If one is
  found, the server will run the update on its own. In essence, all `napari` has to do is run
  `napari-updater` on a separate, detached process every now and then.

The governance of `napari-updater` part will still fall under the `napari` organization (much like
`magicgui`) but it will be usable outside of the napari project, following the same philosophy we
adopted for the constructor work.

#### Tasks

* [ ] Start a simple CLI tool to prototype the role of `napari-updater`, initially under the
      `napari/packaging` repo. This will contain the new version detection code as well as the
      logic to install the new napari environment. Note that we will need to provide a way to
      distribute a "frozen" napari environment, identical to the one being bundled in the
      installers, to ensure that every user gets the same napari installation regardless the
      mechanism used (fresh install from downloaded executable, or updated via `napari-updater`).
* [ ] Refactor the prototype into its own separate project and evolve its feature set to satisfy
      the community feedback (this might include adding a UI, environment management tools,
      installation diagnostics, plugin "co-installability" analysis, etc).

### Milestone 5: Deprecating Briefcase-based installers

See the corresponding section in "Detailed description".

## Backward Compatibility

Migrating the backend on which we build the installers has little impact on the end users who
just want to run napari. That said, `constructor` does not support the AppImage format for
Linux or DMG for macOS, which were the ones previously used with Briefcase. We don't see this
as a problem though, given the small number of downloads each format enjoyed in previous
releases [^napari-releases-json].

It's very important that napari users can still rely on PyPI packages to install plugins. While
conda packaging offers a series of benefits, it can also constitute an access barrier for some
developers and users. For that reason, PyPI packages will still be available on the plugin manager
as an alternative installation method. To enable this mode, the user will need to accept a warning
that details the potential problems it can cause, and how to use the `napari-updater` tool to fix
it, if needed.

## Future Work

The tool handling the updates and managing different napari installations will start simple. In
the future, if it makes sense, we can talk about adding a UI on top.

## Alternatives

> Please refer to the "Related work" section for details on other milestones.

For Milestone 4 "Enabling in-app napari version updates", we considered other options before
deciding to use the currently proposed one. Namely:

* Each napari installation only contains a single conda environment and version. Users can update
  by downloading the newer installer, possibly after having received a notification in a running
  napari instance. This was discarded because it required too many user actions to succeed.
* Each napari installation contains several environments, one per napari version. Each napari
  version can prompt the creation of a new environment in the same base installation if a new
  update is available. Discarded because the update logic could change across napari versions,
  potentially causing issues over time; e.g. old versions are not up-to-date with the latest
  packaging policies established by the napari community.

## Discussion

- [Issue #1001](https://github.com/napari/napari/issues/1001) (Plugin dependency management)
- [PR #4404](https://github.com/napari/napari/pull/4404) (Switch to a more declarative configuration
  for conda packaging)
- [PR #4519](https://github.com/napari/napari/pull/4519) (Initial draft of this NAP and discussion)
- [PR #4602](https://github.com/napari/napari/pull/4602) (PR to discuss the approval of this NAP)
- [Zulip thread](https://napari.zulipchat.com/#narrow/stream/322105-naps/topic/Proposal.20to.20accept.20NAP-2)
  to discuss the approval of this NAP

## References and Footnotes

* Napari on PyPI [^pypi-napari]
* Napari on conda-forge [^napari-feedstock]
* Installation analysis of all napari plugins [^installability-notebook]
* Scientific packaging glossary [^glossary]

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^cc0]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^cc0by].

<!-- Links -->

[^staged-recipes-napari]: <https://github.com/conda-forge/staged-recipes/pull/9983>

[^napari-feedstock-creation]: <https://github.com/conda-forge/napari-feedstock/commit/815a24cadb9522f4fc81a41c9eb89d45b2e284eb>

[^pypi-napari]: <https://pypi.org/project/napari/>

[^napari-feedstock]: <https://github.com/conda-forge/napari-feedstock>

[^briefcase]: <https://beeware.org/project/projects/tools/briefcase/>

[^cibuildwheel]: <https://github.com/pypa/cibuildwheel>

[^staged-recipes-all-plugins]: <https://github.com/conda-forge/staged-recipes/pulls?q=is%3Apr+author%3Agoanpeca+created%3A2022-01-01..2022-05-01>

[^constructor]: <https://github.com/conda/constructor>

[^constructor-upstream]: <https://github.com/napari/napari/issues/4248#issuecomment-1066574137>

[^menuinst]: <https://github.com/conda/menuinst>

[^napari-channel]: <https://anaconda.org/napari/>

[^briefcase-workaround-example]: <https://github.com/napari/napari/blob/be43e127a079f999d830c457d6d69b7c2b56875d/napari/plugins/__init__.py#L34-L42>

[^briefcase-utilities-example]: <https://github.com/napari/napari/blob/be43e127a079f999d830c457d6d69b7c2b56875d/napari/utils/misc.py#L49>

[^pyoxidizer]: <https://pyoxidizer.readthedocs.io/en/stable/>

[^nuitka]: <https://nuitka.net/>

[^napari-releases-json]: <https://api.github.com/repos/napari/napari/releases>

[^mne-constructor]: <https://twitter.com/mne_news/status/1506212014993162247>

[^appimage-crash]: <https://github.com/napari/napari/issues/3487>

[^appimage-crash2]: <https://github.com/napari/napari/issues/3816>

[^napari-packaging-docs]: <https://napari.org/developers/packaging.html>

[^napari-hub-api]: https://api.napari-hub.org/plugins

[^vscode-extensions-ui]: https://code.visualstudio.com/docs/editor/extension-marketplace

[^in-app-update-pr]: https://github.com/napari/napari/pull/4422

[^installability-notebook]: https://colab.research.google.com/drive/1QxbBZYe9-AThGuRsTfwYzT72_UkamXmk

[^glossary]: https://jaimergp.github.io/scientific-packaging-glossary/

[^briefcase-python]: https://github.com/beeware?q=Python+support&type=all&language=&sort=

[^pypi-parallelism-abi]: https://twitter.com/ralfgommers/status/1517410559972589569

[^release-guide]: https://napari.org/developers/release.html

[^scijava-pinnings]: https://github.com/scijava/pom-scijava/blob/ff35ca810a8717c4f461ef24df4986bf1914c673/pom.xml#L307

[^maxiconda]: https://github.com/Semi-ATE/maxiconda-envs

[^conda-forge-pinnings]: https://github.com/conda-forge/conda-forge-pinning-feedstock/blob/32f93dd/recipe/conda_build_config.yaml

[^audithwheel]: https://github.com/pypa/auditwheel

[^delocate]: https://github.com/matthew-brett/delocate

[^delvewheel]: https://github.com/adang1345/delvewheel

[^conda-pip-issue-comment]: https://github.com/napari/napari/issues/3223#issuecomment-972189348

[^pamba]: https://github.com/tlambert03/pamba

[^napari-installer-class]: https://github.com/napari/napari/blob/5c10022337601f350ad64ce56eddf6664306e40e/napari/_qt/dialogs/qt_plugin_dialog.py#L64

[^cc0]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^cc0by]: <https://dancohen.org/2013/11/26/cc0-by/>
