import napari

v = napari.Viewer()

v.scale_bar.visible = True
v.scale_bar.box = True
v.scale_bar.gridded = True

v.text_overlay.visible = True
v.text_overlay.text = 'Points'
v.text_overlay.font_size = 20
v.text_overlay.position = 'bottom_right'

v.grid.enabled = True
v.grid.stride = 2

ll = v.open_sample('napari', 'lily')

napari.run()