# napari's application model

```{important}
**This is not a part of the public napari interface!**

This page is mostly aimed at developers who are interested in contributing to
or understanding the inner workings of napari.
```

```{warning}
**Work in progress!** 
This document is here to give a little guidance on the current
vision for how this application model might be used. It will very likely change
as we develop it.
```

## App-model

The global application singleton can be retrieved with `napari._app_model.get_app()`.  It is an instance of
[`app_model.Application`](https://app-model.readthedocs.io/en/latest/application/).
`app-model` is a Python package that provides a declarative schema for an
application.  It is an abstraction developed by napari developers, with the
needs of napari in mind, but it is agnostic to napari itself (i.e. it should be
reusable by any python GUI application).

Currently, the primary purpose of the `app` is to compose the various
[registries](https://app-model.readthedocs.io/en/latest/registries/) (commands,
keybindings, menus, etc...) into a single name-spaced object.  

## Commands

Commands represent callable objects that "do something" in napari, and usually
have a corresponding representation somewhere in the GUI (such as a button, a
menu item, or a keybinding).

All commands have a string id (e.g. '`napari:layer:duplicate'`), which should be
declared as a member of the `napari._app_model.constants.CommandId` enum.  Internally,
an instance of this enum should be used instead of the string literal when
referring to a command, as it is easier to refactor and test.

```{note}
Some of these command strings MAY be exposed externally in the future. For example, a plugin may wish to refer to a napari command.
```

### Commands should *not* be confused with the public napari API

While it is conceivable that plugins might need/want to refer to one of these
napari commands by its string id, it is **not** currently a goal that napari
end-users would execute any of these commands by their ID.  There should always
be a "pure python" way to import a napari object and call it.  Commands mostly
serve as a way to reference some functionality that needs to be exposed in the
GUI.

## Menus

All napari menus will also have a string id (e.g. `'napari/file'`, or
`'napari/layers/context'`), which should be declared as a member of the
`napari._app_model.constants.MenuId` enum.  Internally, an instance of this enum
should be used instead of the string literal when referring to a menu.

Menus are not limited to the visible menus in the application mene bar, they can
appear anywhere, such as the layerlist context menu that appears when a user
right-clicks on the layer list. They may also be used to create a toolbar (a set
of clickable buttons that execute a command). For example, the mode buttons in
each of the layer controls could be represented as menus with a set of commands.

One of the benefits of the abstraction provided by `app-model` is that actual Qt
Menu objects become simple to construct:

```python
>>> from app_model.backends.qt import QModelMenu
>>> from napari._app_model.constants import MenuId

# create a QMenu with all of the commands registered in the
# layerlist context menu
>>> menu = QModelMenu(menu_id=MenuId.LAYERLIST_CONTEXT, app='napari')

>>> print([i.text() for i in menu.actions()])
[
    'Toggle visibility',
    '',
    'Convert to Labels',
    'Convert to Image',
    'Convert data type',
    '',
    'Duplicate Layer',
    'Split Stack',
    'Split RGB',
    'Merge to Stack',
    'Projections',
    '',
    'Link Layers',
    'Unlink Layers',
    'Select Linked Layers'
]
```

For details on how menus are grouped into sections, and how commands are ordered
within each section, see the `group` and `order` attributes in the
[`app_model.types.MenuRule`
documentation](https://app-model.readthedocs.io/en/latest/types/#app_model.types.MenuRule)

## Keybindings

`app-model` has an [extensive internal representation](https://app-model.readthedocs.io/en/latest/keybindings/) of Key codes, and
combinations of key press events (including *chorded* key press sequences such
as `Cmd+K` *followed by* `Cmd+M`).

We don't yet use them internally, but they will provide independence from
vispy's key codes, and have a nice `IntEnum` api that allows for declaration of
keybindings in namespaced way that avoids usage of strings:

```python
>>> from app_model.types import KeyCode, KeyMod

>>> ctrl_m = KeyMod.CtrlCmd | KeyCode.KeyM

>>> ctrl_m
<KeyCombo.CtrlCmd|KeyM: 2078>
```

## Actions

The "complete" representation of a command, along with its optional placement
in menus and/or keybinding associations is defined by the
[`app_model.types.Action`](https://app-model.readthedocs.io/en/latest/types/#app_model.types.Action)
type.  It composes an
[`app_model.types.CommandRule`](https://app-model.readthedocs.io/en/latest/types/#app_model.types.CommandRule),
[`app_model.types.MenuRule`](https://app-model.readthedocs.io/en/latest/types/#app_model.types.MenuRule)
and
[`app_model.types.KeyBindingRule`](https://app-model.readthedocs.io/en/latest/types/#app_model.types.KeyBindingRule).


The following code would register a new "Split RGB" command, to be added to a
specific section of the layerlist context menu, with a `Cmd+Alt+T` keybinding.

Note that while strings could be used for `id`, `title`, `menus.id` and
`keybindings.primary`, the usage of enums and constants makes refactoring and
maintenance much easier (and provides autocompletion in an IDE!)

```python
from app_model.types import Action, KeyMod, KeyCode
from napari._app_model.constants import CommandId, MenuId, MenuGroup
from napari._app_model.context import LayerListContextKeys as LLCK
from napari._app_model import get_app


# `layers` will be injected layer when this action is invoked
def split_rgb_layer(layers: 'LayerList'):
    ...


action = Action(
    id=CommandId.LAYER_SPLIT_RGB,
    title=CommandId.LAYER_SPLIT_RGB.title,
    callback=split_rgb_layer,
    menus = [
        {
            'id': MenuId.LAYERLIST_CONTEXT,
            'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
            'when': LLCK.active_layer_is_rgb,
        }
    ],
    keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyT }]
)

get_app().register_action(action)
```

````{note}
If you're following along in the console, you may see the following error
when executing the above code:

```python
ValueError: Command 'napari:layer:split_rgb' already registered
```

This is because command id's may currently only be registered once, and associated with a single callback (and napari's internal app already used the `CommandId.LAYER_SPLIT_RGB` id). This MAY change in the future if a need arises.
````

## Dependency Injection

A key component of the command infrastructure is "dependency injection",
currently provided by the package
[`in-n-out`](https://github.com/tlambert03/in-n-out) (which spun out of an
internal napari module).  `app-model` uses `in-n-out` to inject dependencies into all commands in the `CommandsRegistry`.

```{tip}
Dependency injection is just a fancy word for "giving a function or class something it needs to perform its task".
```

In practice, dependency injection will be performed *internally* by napari (i.e.
napari will inject dependencies into some internally or externally provided
function, plugins/users don't use the `@inject` decorator themselves), and the pattern will look something like this:

A user/plugin provides a function

```python
# some user provided function declares a need 
# for Points by using type annotations.
def process_points(points: 'Points'):
    # do something with points
    print(points.name)
```

Internally, napari registers a set of "provider" and "processor" functions in
the `get_app().injection_store`

```python
from napari._app_model import get_app

# return annotation indicates what this provider provides
def provide_points() -> Optional['Points']:
    import napari.viewer
    from napari.layers import Points

    viewer = napari.viewer.current_viewer()
    if viewer is not None:
        return next(
            (i for i in viewer.layers if isinstance(i, Points)), 
            None
        )

get_app().injection_store.register_provider(provide_points)
```

This allows both internal and external functions to be injected with these
provided objects, and therefore called *without* parameters in certain cases.
This is particularly important in a GUI context, where a user can't always be
providing arguments:

```python
>>> injected_func = get_app().injection_store.inject(process_points)
```

```{tip}
The primary place that this injection occurs is *in* `app-model`: in the `run_injected` property of all registered commands in the `CommandsRegistry`. 
```

Note: injection doesn't *inherently* mean that it's always safe to call an
injected function without parameters. In this case, we have no viewer and no
points:

```python
>>> injected_func()

TypeError: After injecting dependencies for NO arguments,
process_points() missing 1 required positional argument: 'points'
```

Our provider was context dependent... Once we have an active viewer with a
points layer, it can provide it:

```python
>>> viewer = napari.view_points(name='Some Points')

>>> injected_func()
Some Points
```

The fact that `injected_func` may now be called without parameters allows it to
be used easily as a command in a menu, or bound to a keybinding.  It is up to
`napari` to determine what providers it will make available, and what type hints
plugins/users may use to request dependencies.

## Motivation & Future Vision

While it's certainly possible that there will be cases where this abstraction proves to be a bit more annoying than the previous "procedural" approach, there are a number of motivations for adopting this abstraction.

1. It gives us an abstraction layer on top of Qt that will make it much easier to explore different application backends (such as a web-based app, etc..)
1. It's easier to test: `app-model` can take care of making sure that commands, menus, keybindings, and actions are rendered, updated, and triggered correctly, and napari can focus on testing the napari-specific logic.
1. It's becomes **much** easier to add & remove contributions from plugins if our internal representation of a command, menu, keybinding is similar to the schema that plugins use. The previous procedural approach made this marriage much more cumbersome.
1. **The Dream**: The unification of napari commands and plugin commands into a registry that can execute commands in response to user input provides an excellent base for "recording" a user workflow.  If all GUI user interactions go through dependency-injected commands, then it becomes much easier to export a script that reproduces a set of interactions.
