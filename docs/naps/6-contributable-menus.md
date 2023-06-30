(nap-6-contributable-menus)=

# NAP-6 — Contributable Menus

```{eval-rst}
:Author: Draga Doncila Pop <ddoncila@gmail.com>
:Created: 2022-12-02
:Resolution: TBD
:Resolved: TBD
:Status: Draft
:Type: Standards Track
:Version effective: TBD
```

## Abstract

Since the initial release of `npe2` infrastructure has been in place for
plugin developers to declare **menu contributions**. These contributions
add new items to menus in napari that have been deemed **contributable**.

Until now, only the layer context menu (available through right clicking on
a layer in the layer list) has been contributable, but much discussion 
has occurred on the list of menus we wish to open up for contribution and 
the guiding principles behind the organization of this list. 

This NAP defines an overall structure for contributable menus, an initial
list of contributable menus that are to be opened up for plugin developers,
and a process for users and plugin developers to propose new contributable
menus to be added to the existing list.

## Motivation and Scope

<!-- This section describes the need for the proposed change. It should describe
the existing problem, who it affects, what it is trying to solve, and why.
This section should explicitly address the scope of and key requirements
for the proposed change. -->

Currently plugin developers can provide processing and analysis extensions strictly through 
**dock widget** contributions. These are exposed to the user under the `Plugins` menu, either
directly at the top level if a plugin only provides a single widget, or in a submenu labelled
with the plugin's display name when a plugin provides multiple widgets.

This `Plugins` menu quickly becomes difficult to parse with increasing number of plugins
installed in an environment, and does not provide sufficient structure for a user to be
able to quickly and coherently navigate through the extensions available to them. 

![A napari viewer with many plugins installed quickly becomes unwieldy.](./_static/napari-many-plugins.png)

The vast majority of plugins (217 out of 263) available today provide at least one `widget` contribution. 
Of these, 140 provide just a single widget 19 provide more than five widgets and 9 provide more than 10. 
This means that while it is important to provide structure within an individual plugin's widgets, 
we must also provide cross-plugin structure so that users with many plugins installed can find widgets 
by the action they want to perform, rather than by hunting across 
endless plugin submenus, or attempting to discern what a plugin's widget might do from its title.

Without meaningful places to put their contributions, plugin developers are coming
up with their own way to organize contributions, whether through numbering widgets in 
their menu, mangling names to achieve a certain order, or coming up with their own unsupported
solutions for adding new menu items. 

The goal of this NAP, therefore, is to provide a structured set of contributable menus
that is easy to navigate, semantically organized and intuitive for both users and plugin developers.

```{note}
It is highly likely that with growing numbers of plugins, widgets and menus, menu navigation
itself becomes more burdensome when hunting for a specific action. Searchability of menu items is 
not within scope for this NAP, but will be made available to users via a command palette.
```

### What is a Menu Contribution?

A `MenuItem` contribution in the [`npe2` manifest](https://github.com/napari/npe2/blob/main/npe2/manifest/contributions/_menus.py) 
adds a new item to one of the `napari` menus (defined by an ID). When this item is clicked, 
the associated `command` is executed. Additionally, `enablement` clauses can be defined 
that control when this menu item is available for clicking and when it is disabled. Similarly, 
a `when` clause can be used to control whether the menu item is visible in the menu at all.

In addition to the menu items themselves, `Submenu` contributions can also be defined,
which add a new submenu to a contributable menu which can be populated with new
`MenuItem` contributions.

Currently, the only `napari` menu to which items can be contributed in this way is the 
layer context menu, accessible by right clicking on a layer in the `LayerList` as shown
in the screenshot below. The new menu items and submenu are produced by the following code snippet:

```yaml
name: napari-demo
display_name: Demo plugin

contributions:
  commands:
    - id: napari-demo.menu_item
      title: A new menu item
      python_name: napari_demo:menu_item
    - id: napari-demo.submenu_item
      title: A new item in a submenu
      python_name: napari_demo:submenu_item

  menus:
    napari/layers/context:
      - submenu: context_submenu
      - command: napari-demo.submenu_item
    hello_world:
      - command: napari-demo.menu_item

  submenus:
    - id: context_submenu
      label: A new submenu
```

This NAP proposes new menu IDs and new top level menus to open for contribution.

![Right click layer context menu with new menu item and submenu contributed by a plugin](./_static/layer-context-menu.png)


### What do Menu Contributions do?

`MenuItem` contributions can be thought of as auxiliary contributions that
provide a dispatch mechanism for other existing contributions. Currently
these would strictly be `widget` contributions, but this mechanism
can easily be extended to other commands, which can take as input 
`napari` objects like specific layers, or the `Viewer`, and produce
output the `Viewer` uses - currently this would be new layers.

Moving forward, new contribution types could be defined that allow
plugin developers to run context aware commands that interact with 
different `Viewer` components without the need for a `widget`.

For example, `LayerEditor` contributions could take the currently
active/selected `Layers` and edit the underlying data, while `LayerGenerator`
contributions could take the same input and create new layers in
the viewer. By providing dedicated contributions for such actions,
`napari` can enforce rules about layer editing and layer 
generation more strictly than for widget contributions,
which, if taking the `Viewer`, can perform arbitrary actions upon
all objects within the `Viewer`.

We therefore propose a menu structure that would be easily
extensible with these new contribution types and provide intuitive
locations for both plugin developers to add their functionality,
and users to find it. 

## Detailed Description

<!-- This section should provide a detailed description of the proposed change. It
should include examples of how the new functionality would be used, intended
use-cases, and pseudocode illustrating its use. -->

We propose an initial set of contributable menus organized by the napari object
being acted upon by the actions within the menu, and the likely output of those 
actions. 

### The `Layers` Menu
Currently the foremost example of such an object is the napari `Layers`, and this
menu therefore contains five submenus organized by the types of processing
the user may wish to perform on the selected `Layer` or `Layers`.

The `Layers` submenus are organized to give the user an immediate
feeling of what might happen to their `Layers` as a result of clicking
one of these menu items.

1. `Visualization` - Items in this submenu allow you to generate visualizations from selected layer or layers.
They do not change the layer data.
2. `Measure` - Items in this submenu provide utilities for summarising information about your layer's data.
3. `Edit` - The items in this submenu change the data of your layer through `Filters` or `Transformations`. 
Additionally `Annotation Tools` provide a location for convenience layer editing tools e.g. `Labels` split/merge actions.
Items in this submenu **should not** generate new layers, but rather act upon the existing layer data.
3. `Generate` - Items in this submenu are the main *analysis* actions you can take on your layer. 
These items should add new layers to the viewer based on analyses and processing of data in your selected layer(s). 
The five proposed submenus are `Projection`, `Segmentation`, `Classification`, `Registration` and `Tracks`.

Many of the actions in this menu exist in the right click layer context menu. These items
should be replicated in the `Layers` menu as needed, both to aid discoverability and
to ensure users are not met with empty menus on initially opening `napari`. It's
also possible that exceedingly common operations e.g. thresholds, are provided in the future
by `napari` itself.

### The `Acquisition` Menu

In addition to the `Layers` menu, we add `Acquisition` as a top level menu.

`Acquisition` will contain widgets and utilities for interfacing with microscopes and
other types of cameras.

### The I/O Utilities Menu

A cursory analysis of widget names revealed a minimum of 17 plugins provide
widgets dedicated to importing and exporting of data, features, models and/or
other material supporting analysis.

These widgets usually require more choices from the user than are currently
possible via the `reader` and `writer` interfaces. Although many discussions have
been raised about expanding the opening and saving options in `napari` to 
support more complex choices ([#1637](https://github.com/napari/napari/issues/1637), 
[#2801](https://github.com/napari/napari/issues/2801), 
[#4611](https://github.com/napari/napari/issues/4611), 
[#4882](https://github.com/napari/napari/pull/4882)...),
we are not presently close to providing a unified interface for complex file opening/saving.
Additionally, default `napari` opening and saving is entirely focused on reading data into 
layer, but there are many other reasons a user may wish to read a file or save some output
from the `Viewer`. 

It is likely, therefore, that some plugins will always have bespoke interfaces for importing and
exporting various file formats. These interfaces will be exposed via the new `File->I/O Utilities`
menu.

### Plugin Submenus

The goal of the newly proposed menus is to provide a natural place where generally applicable
actions can be semantically organized and easy to locate. However, many `napari` plugins
contain an assortment of highly specialized widgets (that often interact with each other)
that support highly specific, and sometimes ordered, workflows and analyses.

It may never make sense for such plugins to distribute their widgets across the different
`napari` menus, particularly when they are designed to work in concert on specific
data formats or layer types.

We therefore give plugin developers full control over their own submenu under `Plugins->My Plugin`.
Plugin developers can organize all contributions under this submenu as they see fit, including adding their own submenus of arbitrary depth.

### Complete Set of Proposed Contributable `napari` Menus

```
File
├─ ...
├─ IO Utilities
Layers
├─ Visualization
├─ Edit
│  ├─ Annotation Tools
│  ├─ Filter
│  ├─ Transform
├─ Measure
├─ Generate
│  ├─ Registration
│  ├─ Projection
│  ├─ Segmentation
│  ├─ Tracks
│  ├─ Classification
Acquisition
```

As a case study, we take four plugins offering between 9 and 14 widget contributions and arrange their widgets in these menus: 
`empanada-napari`, `napari-stracking`, `napari-mm3` and `napari-clemreg`. Where a plugin's widgets don't
naturally fit into one of the proposed menus, they are left in the plugin's own submenu.
Note that we have arranged these widgets purely based on title and cursory inspection of the documentation, 
so this should not be considered a concrete proposal for the structure of these plugins.

```
File
├─ ...
├─ IO Utilities
│  ├─ Export 2D segmentations (empanada-napari)
│  ├─ Store training dataset (empanada-napari)
│  ├─ nd2ToTIFF (napari-mm3)
Layers
├─ Visualization
│  ├─ SFilterTrack (napari-stracking)
├─ Edit
│  ├─ Annotation Tools
│  │  ├─ Merge Labels (empanada-napari)
│  │  ├─ Delete Labels (empanada-napari)
│  │  ├─ Split Labels (empanada-napari)
│  │  ├─ Jump to label (empanada-napari)
│  │  ├─ Find next available label (empanada-napari)
│  │  ├─ Pick training patches (empanada-napari)
│  ├─ Filter
│  ├─ Transform
│  │  ├─ make_image_warping (napari-clemreg)
├─ Measure
│  ├─ SParticlesProperties (napari-stracking)
│  ├─ STracksFeatures (napari-stracking)
├─ Generate
│  ├─ Registration
│  ├─ Projection
│  ├─ Segmentation
│  │  ├─ 2D Inference (empanada-napari)
│  │  ├─ 3D Inference (empanada-napari)
│  │  ├─ make_log_segmentation (napari-clemreg)
│  │  ├─ make_clean_binary_segmentation (napari-clemreg)
│  │  ├─ SegmentOtsu (napari-mm3)
│  │  ├─ SegmentUnet (napari-mm3)
│  │  ├─ Foci (napari-mm3)
│  │  ├─ SDetectorDog (napari-stracking)
│  │  ├─ SDetectorDoh (napari-stracking)
│  │  ├─ SDetectorLog (napari-stracking)
│  │  ├─ SDetectorSeg (napari-stracking)
│  ├─ Tracks
│  │  ├─ SLinkerShortestPath (napari-stracking)
│  │  ├─ Tracks (napari-mm3)
│  ├─ Classification
│  ├─ make_point_cloud_sampling (napari-clemreg)
│  ├─ make_point_cloud_registration (napari-clemreg)
Acquisition
Plugins
├─ empanada-napari
│  ├─ Finetune a model
│  ├─ Train a model
│  ├─ Register a model
│  ├─ Get model info
├─ napari-stracking
│  ├─ SScale
│  ├─ SPipeline
├─ napari-mm3
│  ├─ Compile
│  ├─ PickChannels
│  ├─ Subtract
│  ├─ Annotate
│  ├─ Colors
├─ napari-clemreg
│  ├─ mask_roi
│  ├─ make_data_preprocessing
│  ├─ train_model
│  ├─ predict_from_model
```

### Items that Don't Fit?

Where a plugin developer feels that none of the submenus of a given menu are suitable
for their purpose, they should add their item to the deepest applicable menu. For
example, a widget that takes a layer and produces a new layer through random perturbations
would not fit under `Layers -> Generate -> Segmentation` but could fit under 
`Layers -> Generate`.

Where a plugin developer feels no top level menu or submenu is suitable for their purpose,
they should add their item to their own plugin's submenu under `Plugins -> Your Plugin`,
and consider requesting a new contributable menu via the process described below.

The top level menu bar in `napari` is not open for contribution, and new top level
menus can only be added via the process described below.

### Process for Expanding Set of Contributable Menus

New contributable menus or submenus will be added either following periodic analysis
of all plugin contributions, or following user request upon core developer consensus.

#### Periodic Contribution Analysis

As the number of plugins, types of contributions and `Viewer` interactions grows,
it is important that the set of contributable menus is periodically assessed
to add new submenus as required.

Every 6 months to 1 year, the core developers will perform an analysis on 
the total set of menu item and widget contributions of all plugins, and 
derive new groupings to ensure that the length of each submenu remains 
managable. 

For example, consider the `Layers -> Generate -> Segmentation` menu. If 
analysis of the plugin ecosystem reveals 40 different contributions
for Watershed Segmentation, a new `Watershed` submenu would be added
under `Layers -> Generate -> Segmentation -> Watershed`.

#### User Request

`napari` users or plugin developers can at any moment raise an issue
on the `napari` repository to open up an existing menu for contribution
or to add a new submenu to any of the currently contributable menus.

Core developers will assess the proposed menu/submenu based on its 
generality, the number of existing plugins that may contribute
to this menu, whether the proposed menu is sufficiently meaningful
to be immediately understandable by users and other plugin developers,
and whether there is significant overlap with existing contributable
menus.

Once core developer consensus on adding the menu item is achieved,
a Pull Request will be raised opening up this menu for contribution.
The user proposing the menu is not responsible for opening this Pull
Request, though they may, if they wish.

## Related Work

<!-- This section should list relevant and/or similar technologies, possibly in
other libraries. It does not need to be comprehensive, just list the major
examples of prior and relevant art. -->

The [`napari-tools-menu`](https://github.com/haesleinhuepf/napari-tools-menu) 
proposes one top level menu wherein all menu items and submenus are organized.

[Fiji/ImageJ](https://imagej.net/plugins/) have their own structure for their 
top level `Plugins` menu.

## Implementation

<!-- This section lists the major steps required to implement the NAP. Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted. Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this NAP
should be linked to from here. (A NAP does not need to be implemented in a
single pull request if it makes sense to implement it in discrete phases).

If a new NAP document is created, it should be added to the documentation Table
of Contents as an item on `napari/docs/_toc.yml`. -->

- Add list of menus to `napari/constants/_menus.py`, warning for invalid IDs and new top level menus as per [#5153](https://github.com/napari/napari/pull/5153/files/e55ed333866384d85a780b8c2ab1591fab460300#diff-4ab84b59ab3d5cadbcf97fb21bd6547982d5b444251fd627a0b50deb51bec83b)
- Implement menu contribution interface that allows plugins to refer to their own plugin submenu
- Add functionality for opening widgets from a menu item
- Expose menu contribution reference in `npe2`
- Write menu contribution guide in `npe2`

## Backward Compatibility

<!-- This section describes the ways in which the NAP affects backward
compatibility, including both breakages and decisions that better support
backward compatibility. -->

This work does not have any backward compatibility considerations for
existing features.

In future, backward compatibility concerns could arise when a menu name/ID is
changed in `napari`, or when `napari` removes a menu that was previously 
contributable.

In the first case, no change is required in plugin manifests, as `napari`
can simply maintain logic for migrating old IDs to new ones.

If a contributable `napari` submenu is removed, this should be highlighted
first by a deprecation warning. Once the submenu is deprecated, contributions
that refer to this submenu should instead be placed in the higher level
menu. For example, if `Layers -> Generate -> Segmentation` is removed, 
existing contributions referring to this ID will be placed under
`Layers -> Generate`.

If a contributable `napari` top level menu is removed, this should be highlighted
first by a deprecation warning. Before the submenu is deprecated, core developers
must work to identify plugins that will need migration and aid the migration
process by opening issues and PRs as required. Top level menus should not be removed
without a clear migration guide of where these contributions should be placed in 
the future. After the menu is deprecated, contributions referring to this menu
should raise a warning, and be placed in the plugin's own submenu at the highest
level.

Plugin contributions to non contributable menus will raise warnings and be placed
in the plugin's submenu at the top level.

## Future Work

As mentioned above, a key feature to support rapid browsing for actions is the
search functionality via the command palette. This is actively being worked
on and is essential for navigation.

Once more contribution types are exposed for users, it's important that users
are aware why certain actions are disabled when the user doesn't meet the
requisite context declared in the contribution's `enablement` clause. Since
the syntax for declaring these contexts is strictly defined, we should be
able to surface information to suers about what is required for the action
to be enabled and functional. For example, an action could declare itself
enabled only when a points layer *and* an image layer are selected. If 
the user has only selected an image layer, we could indicate the missing
context to the user e.g. "Action takes a points and image layer, but no
points layer is selected". 

A desired attribute of these menu items is that users always know what
will happen when they click a menu item. Does a widget open? Is the layer
edited? Is a new layer added to the `Viewer`? Once more contribution 
types are exposed, we should be able to either add this information 
as metadata in the manifest file, or infer it from return type 
annotations of contribution commands, and also expose this to the user.

Finally, the number of actions in each menu is heavily dependent on
the plugins installed in the user's environment. Given a complete set of
contributable menus, we could dynamically inspect how many menu items
each submenu contains, and group them appropriately for the user while
limiting unnecessary depth. For example, if the user's environment has
six plugins installed that each provide a `Watershed` segmentation,
we could display a `Layers -> Generate -> Segmentation -> Watershed`
submenu. If the user has just one `Watershed` segmentation plugin 
installed, this submenu would not appear. This would require very careful 
design to ensure the user still knows what to expect when they load up
the `Viewer`.

## Alternatives

The main alternative is the proposed `Tools` menu from [npe2 #161](https://github.com/napari/npe2/pull/161).

This is a single top level menu containing the same submenus as our proposed list, but organized
roughly in order of when actions may be performed in a standard image processing workflow.

General feedback from the community and the core developers is that this menu structure, while mostly
containing individual submenus that make sense:

- is too long and therefore difficult to parse at a glance
- does not give the user a good indication of what inputs an action takes and what its output will be
- is not semantically structured and is rather just a one stop shop for "plugin stuff"
- will be difficult to extend further in meaningful ways as we develop more complex viewer
interactions and plugin contributions e.g. multi canvas


## Discussion

- **[May 8 2022: npe2 #160](https://github.com/napari/npe2/pull/160)** is opened and merged during core dev hackathon. Allows arbitrary menu locations in npe2 to support plugins contributing to other plugins, etc. Validation would happen elsewhere.
- **[May 8 2022: npe2 #161](https://github.com/napari/npe2/pull/161)** is opened with almost instant approvals. Initial feedback is that it's difficult for people to know the input/output of menu items, suggests creating a NAP. Complexity arises with desire to declare contributable menus but still allow plugins to contribute to other plugin's menus.
- **Jun 2 2022: npe2 #161** After further discussion (on zulip and in PR), this schema is identified as potentialy too limiting and there is mention that #160 may need to be reverted. A NAP is once again suggested as this is an influential decision with lots of opinions.
- **Jun 13 2022: npe2 #161** is closed and #160 is reverted, with comment for follow up over in the napari repo.
- **[Sep 30 2022: napari #5153](https://github.com/napari/napari/pull/5153)** opened with same list as in npe2, minimal discussion and input.
- **Oct 28 2022: napari #5153** discussion on core devs zulip stream begins. Developers mostly agree on the inidividual menu items but don't like how deep the `Tools` menu already is, and the lack of semantic meaning in its structure. 

## References and Footnotes

All NAPs should be declared as dedicated to the public domain with the CC0
license [^id3], as in `Copyright`, below, with attribution encouraged with
CC0+BY [^id4].

[^id3]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^id4]: <https://dancohen.org/2013/11/26/cc0-by/>

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^id3]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^id4].
