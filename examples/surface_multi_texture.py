"""
Surface with multiple textures
==============================

This example demonstrates one possible method for displaying a 3D surface with
multiple textures.

Thanks to Emmanuel Reynaud and Luis Gutierrez for providing the gorgeous coral
model for this demo. You can find the data on FigShare:
https://doi.org/10.6084/m9.figshare.22348645

More information on the methods used to generate this model can be found in *L.
Gutierrez-Heredia, C. Keogh, E. G. Reynaud, Assessing the Capabilities of
Additive Manufacturing Technologies for Coral Studies, Education, and
Monitoring. Front. Mar. Sci. 5 (2018), doi:10.3389/fmars.2018.00278.*

A bit about 3D models
---------------------

A standard way to define a 3D model (mesh, or Surface in napari) is by listing
vertices (3D point coordinates) and faces (triplets of vertex indices - each
face is a triangle in 3D space).

In some cases, the color of a vertex is given by a single point value that is
then colormapped on the fly. In other cases, each vertex or face may be
assigned a specific color. These methods are demonstrated in
:ref:`sphx_glr_gallery_surface_texture_and_colors.py`.

In the case of "photorealistic" models, the color of each vertex is instead
determined by mapping a vertex to a point in an image called a texture using 2D
texture coordinates in the range [0, 1]. The color of each individual pixel is
smoothly interpolated (sampled) on the fly from the texture (the GPU makes this
interpolation very fast).

Napari does not (yet) support models with multiple textures or materials.
If the textures don't overlap, you can display them on separate meshes as shown
in this demo.

If the textures do overlap, you may instead be able to combine the textures as
images. This relies on textures having the same texture coordinates, and may
require resizing the textures to match each other.

.. tags:: visualization-nD
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pooch
from vispy.io import imread, read_mesh

import napari

###############################################################################
# Download the model
# ------------------
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
    if not file_name.exists():
        print(f"downloading {file_name}")
        download(
            f"doi:{doi}/{file_name}",
            output_file=tmpdir / file_name,
            pooch=None,
        )
    else:
        print(f"using cached {tmpdir / file_name}")

###############################################################################
# Load the model
# --------------
# Next, the model data fron the wavefront (.obj) file. Currently napari/vispy
# do not support reading material properties (wavefront .mtl files) nor
# separate texture and vertex indices (i.e. repeated vertices). Normal vectors
# read from the file are also ignored and re-calculated from the faces.
vertices, faces, _normals, texcoords = read_mesh(tmpdir / data_files["mesh"])

###############################################################################
# Load the textures
# -----------------
# This model comes with two textures: `Texture_0` is generated from
# photogrammetry of the actual object, and `GeneratedMat2` is a generated
# material to fill in  parts of the model lacking photographic texture. The
# texture images need to be flipped because of `how OpenGL expects the texture
# data to be ordered in memory
# <https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml>`_:
#
#   The first element corresponds to the lower left corner of the texture
#   image. Subsequent elements progress left-to-right through the remaining
#   texels in the lowest row of the texture image, and then in successively
#   higher rows of the texture image. The final element corresponds to the
#   upper right corner of the texture image.
photo_texture = np.flipud(imread(tmpdir / data_files["Texture_0"]))
generated_texture = np.flipud(imread(tmpdir / data_files["GeneratedMat2"]))

###############################################################################
# This is what the texture images look like in 2D:
fig, axs = plt.subplots(1, 2)
axs[0].set_title(f"Texture_0 {photo_texture.shape}")
axs[0].imshow(photo_texture)
axs[0].set_xticks((0, photo_texture.shape[1]), labels=(0.0, 1.0))
axs[0].set_yticks((0, photo_texture.shape[0]), labels=(1.0, 0.0))
axs[1].set_title(f"GeneratedMat2 {generated_texture.shape}")
axs[1].imshow(generated_texture)
axs[1].set_xticks((0, generated_texture.shape[1]), labels=(0.0, 1.0))
axs[1].set_yticks((0, generated_texture.shape[0]), labels=(1.0, 0.0))
fig.show()

###############################################################################
# Create the napari layers
# ------------------------
# Next create two separate layers with the same mesh - once with each texture.
# In this example the texture coordinates happen to be the same for each
# texture, but this is not a strict requirement.
photo_texture_layer = napari.layers.Surface(
    (vertices, faces),
    texture=photo_texture,
    texcoords=texcoords,
    name="Texture_0",
)
generated_texture_layer = napari.layers.Surface(
    (vertices, faces),
    texture=generated_texture,
    texcoords=texcoords,
    name="GeneratedMat2",
)

###############################################################################
# Add the layers to a viewer
# --------------------------
# Finally, create the viewer and add the Surface layers.
# sphinx_gallery_thumbnail_number = 2
viewer = napari.Viewer(ndisplay=3)

viewer.add_layer(photo_texture_layer)
viewer.add_layer(generated_texture_layer)

viewer.camera.angles = (90.0, 0.0, -75.0)
viewer.camera.zoom = 75

if __name__ == '__main__':
    napari.run()
