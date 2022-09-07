"""
Action manager
==============

.. tags:: gui, experimental
"""
from random import shuffle

import numpy as np
from skimage import data

import napari
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
from napari.components import ViewerModel
from napari.utils.action_manager import action_manager


def rotate45(viewer: napari.Viewer):
    """
    Rotate layer 0 of the viewer by 45º

    Parameters
    ----------
    viewer : napari.Viewer
        active (unique) instance of the napari viewer

    Notes
    -----
    The `viewer` parameter needs to be named `viewer`, the action manager will
    infer that we need an instance of viewer.
    """
    angle = np.pi / 4
    from numpy import cos, sin

    r = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    layer = viewer.layers[0]
    layer.rotate = layer.rotate @ r


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True)

layer_buttons = viewer.window.qt_viewer.layerButtons

# Button do not need to do anything, just need to be pretty; all the action
# binding and (un) binding will be done with the action manager, idem for
# setting the tooltip.
rot_button = QtViewerPushButton(None, 'warning')
layer_buttons.layout().insertWidget(3, rot_button)


def register_action():
    # Here we pass ViewerModel as the KeymapProvider as we want it to handle the shortcuts.
    # we could also pass none and bind the shortcuts at the window level – though we
    # are trying to not change the KeymapProvider API too much for now.
    # we give an action name to the action for configuration purposes as we need
    # it to be storable in json.

    # By convention (may be enforce later), we do give an action name which is iprefixed
    # by the name of the package it is defined in, here napari,
    action_manager.register_action(
        name='napari:rotate45',
        command=rotate45,
        description='Rotate layer 0 by 45deg',
        keymapprovider=ViewerModel,
    )


def bind_shortcut():
    # note that the tooltip of the corresponding button will be updated to
    # remove the shortcut.
    action_manager.unbind_shortcut('napari:reset_view')  # Control-R
    action_manager.bind_shortcut('napari:rotate45', 'Control-R')


def bind_button():
    action_manager.bind_button('napari:rotate45', rot_button)


# we can all bind_shortcut or register_action or bind_button in any order;
# this let us configure shortcuts even if plugins are loaded / unloaded.
callbacks = [register_action, bind_shortcut, bind_button]

shuffle(callbacks)
for c in callbacks:
    print('calling', c)
    c()


# We can set the action manager in debug mode, to help us figure out which
# button is triggering which action. This will update the tooltips of the buttons
# to include the name of the action in between square brackets.

action_manager._debug(True)

# Let's also modify some existing shortcuts, by unbinding a few existing actions,
# and rebinding them with new shortcuts; below we change the add and select mode
# to be the = (same as + key on US Keyboards but without modifiers) and - keys.
# unbinding returns the old key if it exists; but we don't use it.

# in practice you likely don't need to modify the shortcuts this way as it will
# be implemented in settings, though you could imagine a plugin that would
# allow toggling between many keymaps.

settings = {
        'napari:activate_points_add_mode' : '=',
        'napari:activate_points_select_mode': '-',
}


for action, key in settings.items():
   _old_shortcut = action_manager.unbind_shortcut(action)
   action_manager.bind_shortcut(action, key)

if __name__ == '__main__':
    napari.run()
