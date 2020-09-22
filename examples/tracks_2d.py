import napari
import numpy as np

def _circle(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def tracks_2d(num_tracks = 10):
    """ create 3d+t track data """
    tracks = []

    for track_id in range(num_tracks):

        # space to store the track data and properties
        track = np.zeros((100, 6), dtype=np.float32)

        # time
        track[:,0] = np.arange(track.shape[0])

        radius = 20+30*np.random.random()
        theta = track[:,0]*0.1 + np.random.random()*np.pi
        x, y = _circle(radius, theta)

        track[:,1] = 50. + y
        track[:,2] = 50. + x
        track[:,3] = theta
        track[:,4] = radius
        track[:,5] = track_id

        tracks.append(track)


    tracks = np.concatenate(tracks, axis=0)
    data = tracks[:,:3] # just the coordinate data

    properties = {'track_id': tracks[:,5],
                  'time': tracks[:,0],
                  'theta': tracks[:,3],
                  'radius': tracks[:,4]}

    graph = {}
    return data, properties, graph


tracks, properties, graph = tracks_2d(num_tracks=10)

with napari.gui_qt():
    viewer = napari.Viewer()
    # viewer.add_points(tracks, size=1)
    viewer.add_tracks(tracks, properties=properties, graph=graph, name='tracks')
