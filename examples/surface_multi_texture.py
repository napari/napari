"""
Surface with multiple textures
======================================

Display a 3D surface with multiple textures.

Napari does not (yet) support models with multiple textures or materials, but
for separable textures you can display them on mutliple overlayed meshes.

.. tags:: visualization-nD
"""
import os

import numpy as np
import pooch
from vispy.io import imread, read_mesh

import napari

# create the viewer and window
viewer = napari.Viewer(ndisplay=3)

# download the model - thanks Emmanuel Reynaud and Luis Gutierrez!
# https://doi.org/10.6084/m9.figshare.22348645
download = pooch.DOIDownloader(progressbar=True)
doi = "10.6084/m9.figshare.22348645.v1"
tmpdir = pooch.os_cache("napari-surface-texture-example")
os.makedirs(tmpdir, exist_ok=True)
data_files = {
    "mesh": "PocilloporaDamicornisSkin.obj",
    # "materials": "PocilloporaVerrugosaSkinCrop.mtl",  # not yet supported
    "Texture_0": "PocilloporaDamicornisSkin_Texture_0.jpg",
    "GeneratedMat2": "PocilloporaDamicornisSkin_GeneratedMat2.png",
}
print(f"downloading data into {tmpdir}")
for file_name in data_files.values():
    if not (tmpdir / file_name).exists():
        print(f"downloading {file_name}")
        download(
            f"doi:{doi}/{file_name}",
            output_file=tmpdir / file_name,
            pooch=None,
        )
    else:
        print(f"using cached {file_name}")

# load the model data
vertices, faces, _normals, texcoords = read_mesh(tmpdir / data_files["mesh"])

# add the mesh to the viewer once for each texture
texture = np.flipud(imread(tmpdir / data_files["Texture_0"]))
viewer.add_surface(
    (vertices, faces),
    texture=texture,
    texcoords=texcoords,
    name="Texture_0",
)
texture = np.flipud(imread(tmpdir / data_files["GeneratedMat2"]))
viewer.add_surface(
    (vertices, faces),
    texture=texture,
    texcoords=texcoords,
    name="GeneratedMat2",
)

viewer.camera.angles = (90.0, 0.0, -75.0)
viewer.camera.zoom = 75

if __name__ == '__main__':
    napari.run()
