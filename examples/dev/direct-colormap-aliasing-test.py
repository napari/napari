import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import napari

# Set the number of steps
nb_steps = 10000

# Create a dummy label image
base = np.linspace(start=1, stop=nb_steps, num=nb_steps).astype('uint16')
label_img = np.repeat(base.reshape([1, base.shape[0]]), int(nb_steps/10), axis=0)

# Add a harder test case: Randomly order the label values.
# But assign them monotonously increasing feature values (=> if it's off by one, the error is more visible)

shuffled = np.linspace(start=1, stop=nb_steps, num=nb_steps).astype('uint16')
np.random.shuffle(shuffled)
label_img_shuffled = np.repeat(shuffled.reshape([1, shuffled.shape[0]]), int(nb_steps/10), axis=0)
df_shuffled = pd.DataFrame([shuffled, np.linspace(start=1, stop=nb_steps, num=nb_steps).astype('uint16')]).T
df_shuffled.columns = ['label', 'feature']

# calculate the colormaps manually
lower_contrast_limit = 1
upper_contrast_limit = nb_steps

df = df_shuffled

df['feature_scaled_shuffled'] = (
    (df['feature'] - lower_contrast_limit) / (upper_contrast_limit - lower_contrast_limit)
)
colors = plt.cm.get_cmap('viridis')(df['feature_scaled_shuffled'])
colormap_shuffled = dict(zip(df['label'].astype(int), colors))

df['feature_scaled_not_shuffled'] = (
    (df['label'] - lower_contrast_limit) / (upper_contrast_limit - lower_contrast_limit)
)
colors_ordered = plt.cm.get_cmap('viridis')(df['feature_scaled_not_shuffled'])
colormap_ordered = dict(zip(df['label'].astype(int), colors_ordered))

# calculate texel positions as colors for debugging
# uncomment the relevant line in the shader to compare
from napari._vispy.layers.labels import build_textures_from_dict, hash2d_get  # noqa

tex_shape = (1000, 1000)  # NOTE: this has to be equal to the actual texture shape in build_textures_from_dict!
keys, values = build_textures_from_dict(colormap_ordered)
texel_pos_img = np.zeros((1, nb_steps, 4))
texel_pos_img[..., -1] = 1  # alpha
for k in range(nb_steps):
    grid_position = hash2d_get(k + 1, keys, values)[0]
    # divide by shape and set to RG values like in shader (tex coords)
    texel_pos_img[:, k, :2] = (np.array(grid_position) + 0.5) / tex_shape

# Add to napari
viewer = napari.Viewer()
viewer.add_image(label_img, colormap='viridis')
labels_layer_shuffled = viewer.add_labels(label_img_shuffled, opacity=100)
#viewer.add_image(texel_pos_img, rgb=True)
labels_layer_ordered = viewer.add_labels(label_img, opacity=100)
viewer.grid.enabled = True
viewer.grid.shape = -1, 1

# Set the label image colormaps
labels_layer_shuffled.color = colormap_shuffled
labels_layer_ordered.color = colormap_ordered

# TMP debugging stuff
vlab = viewer.window._qt_viewer.layer_to_visual[viewer.layers[-1]]


napari.run()
