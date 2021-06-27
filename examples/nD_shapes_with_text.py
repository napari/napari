from skimage import data
import napari


blobs = data.binary_blobs(
    length=100, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.03
).astype(float)

viewer = napari.view_image(blobs.astype(float), ndisplay=3)

n = 50
shape = [[[n, 40, 40], [n, 40, 60], [n + 20, 60, 60], [n + 20, 60, 40]]]

properties = {'z_index': [n]}
text = {'text': 'z_index', 'color': 'green', 'anchor': 'upper_left'}

shapes_layer = viewer.add_shapes(
    shape,
    edge_color=[0, 1, 0, 1],
    face_color='transparent',
    properties=properties,
    text=text,
)

napari.run()