import napari
from tracks_3d import tracks_3d


if __name__ == '__main__':
    tracks, properties, graph = tracks_3d(num_tracks=2000, track_length=500)
    vertices = tracks[:, 1:]

    viewer = napari.Viewer()
    print('initializing with default track manager ...')
    track_layer = viewer.add_tracks(tracks, properties=properties, name='tracks')
    print('building interactivity graph ...')
    track_layer.editable = True
    print('done')

    napari.run()

    # type in the napari console: viewer.layers['tracks']._manager._is_serialized = False
    # not working properly because the data extent and slicing is forcing serialization
