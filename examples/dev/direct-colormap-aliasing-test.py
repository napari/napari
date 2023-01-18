import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import napari

# Set the number of steps
nb_steps = 15#100 * 70

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

# Add to napari
viewer = napari.Viewer()
viewer.add_image(label_img, colormap='viridis')
labels_layer_shuffled = viewer.add_labels(label_img_shuffled, opacity=100)
labels_layer_ordered = viewer.add_labels(label_img, opacity=100)
viewer.grid.enabled = True

# Set the label image colormaps
labels_layer_shuffled.color = colormap_shuffled
labels_layer_ordered.color = colormap_ordered

napari.run()
