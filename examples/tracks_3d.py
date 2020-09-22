import napari
import numpy as np

def lissajous(t):
    a = np.random.random(size=(3,))*80. - 40.
    b = np.random.random(size=(3,))*0.1
    c = np.random.random(size=(3,))*0.01
    return (a[i]*np.cos(b[i]*t+c[i]) for i in range(3))


def tracks_3d(num_tracks = 10):
    """ create 3d+t track data """
    tracks = []

    for track_id in range(num_tracks):

        # space to store the track data and properties
        track = np.zeros((100, 5), dtype=np.float32)

        # time
        track[:,0] = np.arange(track.shape[0])
        x, y, z = lissajous(track[:,0])

        track[:,1] = 50. + z
        track[:,2] = 50. + y
        track[:,3] = 50. + x
        track[:,4] = track_id

        tracks.append(track)


    tracks = np.concatenate(tracks, axis=0)
    data = tracks[:,:4] # just the coordinate data

    properties = {'track_id': tracks[:,4],
                  'time': tracks[:,0],
                  'x': tracks[:,3],
                  'y': tracks[:,2],
                  'z': tracks[:,1]}

    graph = {}
    return data, properties, graph

tracks, properties, graph = tracks_3d(num_tracks=100)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_points(tracks, size=1)
    viewer.add_tracks(tracks, properties=properties, graph=graph, name='tracks')
