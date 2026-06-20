import skimage.data

import napari

napari.imshow(skimage.data.coins(), name='sample')
napari.run()
