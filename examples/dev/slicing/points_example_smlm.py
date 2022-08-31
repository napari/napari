import napari
import pooch
import csv
import numpy as np

"""
This data comes from the Neurocyto Lab's description of the ThunderSTORM format.
This file format is used to represent single molecule localizations.

With respect to the napari async slicing work, this dataset is small enough that it performs well in synchronous mode.

If someone is interested, then you can use the uncertainty_xy attribute from the STORM data to change the point size.

More information is available here: http://www.neurocytolab.org/tscolumns/
"""

storm_path = pooch.retrieve(
    url='http://www.neurocytolab.org/wp-content/uploads/2018/06/ThunderSTORM_TS3D.csv',
    known_hash='665a28b2fad69dbfd902e4945df04667f876d33a91167614c280065212041a29',
)

with open(storm_path) as csvfile:
    data = list(csv.reader(csvfile))

data = np.array(data[1:]).astype(float)
data = data[:, 1:4]
print('data shape', data.shape, data)

viewer = napari.view_points(data)

napari.run()
