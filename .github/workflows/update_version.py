import tomlkit
import os
from napari import __version__


pyprjct = os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml')

with open(pyprjct, 'r') as f:
    content = f.read()

doc = tomlkit.parse(content)
doc['tool']['briefcase']['version'] = __version__

with open(pyprjct, 'w') as f:
    f.write(tomlkit.dumps(doc))
