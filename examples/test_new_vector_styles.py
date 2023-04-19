
import numpy as np

import napari

viewer = napari.Viewer()

fake_vectors = np.zeros((10,2,4))
fake_vectors[:,0,0] = np.arange(10)
fake_vectors[:,1,2] = np.arange(10)+1

foo = np.arange(10)
properties = {'foo': foo/foo.max()}

viewer.add_vectors(fake_vectors, name='fake_vectors', properties=properties)

napari.run()


