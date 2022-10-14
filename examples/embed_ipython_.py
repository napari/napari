"""
Embed IPython
=============

Start napari and land directly in an embedded ipython console with qt event loop.

A similar effect can be achieved more simply with `viewer.update_console(locals())`,
such as shown in https://github.com/napari/napari/blob/main/examples/update_console.py.

However, differently from `update_console`, this will start an independent
ipython console which can outlive the viewer.

.. tags:: gui
"""

import napari
from IPython.terminal.embed import InteractiveShellEmbed

# any code
text = 'some text'

# initalize viewer
viewer = napari.Viewer()

# embed ipython and run the magic command to use the qt event loop
sh = InteractiveShellEmbed()
sh.enable_gui('qt')  # equivalent to using the '%gui qt' magic
sh()  # open the embedded shell

# From there, you can access the script's scope, such as the variables `text` and `viewer`
