"""
Start napari and land directly in an embedded ipython console with qt event loop
"""
import napari
from IPython.terminal.embed import InteractiveShellEmbed

# any code
text = 'some text'

# initalize viewer
viewer = napari.Viewer()

# embed ipython and run the magic command to use the qt event loop
sh = InteractiveShellEmbed()
sh.run_cell('%gui qt')
sh()  # the ipython shell will open

# From there, you can access the script's scope, such as the variables `text` and `viewer`
