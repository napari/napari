import napari, skimage.data

napari.imshow(skimage.data.coins(), name="sample")
napari.run()
