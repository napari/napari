"""
updates version and requirements in pyproject.toml briefcase settings
"""

import tomlkit
import os
import napari


napari_root = os.path.dirname(napari.__file__)
req_txt = os.path.join(napari_root, '..', 'requirements', 'default.txt')
with open(req_txt) as f:
    reqs = [l.split("#")[0].strip() for l in f if not l.startswith("#")]

pyprjct = os.path.join(napari_root, '..', 'pyproject.toml')

with open(pyprjct, 'r') as f:
    content = f.read()

doc = tomlkit.parse(content)
doc['tool']['briefcase']['version'] = napari.__version__
doc['tool']['briefcase']['app']['napari']['requires'] = reqs + [
    "pip",
    "PySide2==5.14.2",
]

print("patching pyroject.toml to version: ", napari.__version__)
print("patching pyroject.toml requirements to : ", reqs + ["pip"])

with open(pyprjct, 'w') as f:
    f.write(tomlkit.dumps(doc))
