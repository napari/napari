"""
Lat/Lon data with background map
================================

For a given set of points of interest (POI) with lat/lon coordinates,
download a background map from contextily and display it in napari together with the POI.
Requires geopandas and contextily to be installed.

.. tags:: gui
"""

import contextily as ctx
import geopandas as gpd
import pandas as pd

import napari

# some point of interest with lat/lon coordinates and a description
df = pd.DataFrame([
    {'lon': 14.3983569, 'lat': 50.0897206, 'sight': 'old castle', 'nature': False, 'stars': 5.0},
    {'lon': 14.4112958, 'lat': 50.0864922, 'sight': 'crowded bridge', 'nature': False, 'stars': 4.0},
    {'lon': 14.4178942, 'lat': 50.0629778, 'sight': 'even older castle', 'nature': False, 'stars': 4.1},
    {'lon': 14.4495206, 'lat': 50.0884767, 'sight': 'nice view from here', 'nature': True, 'stars': 3.2},
    {'lon': 14.4052019, 'lat': 50.1171367, 'sight': 'zoo', 'nature': True, 'stars': 5.0}
])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')


# convert bounds to crs=3857 (web mercator), and get the background map from contextily
boundsWgs84 = gdf.total_bounds
bounds = gpd.GeoSeries(gpd.GeoDataFrame(geometry=gpd.points_from_xy([boundsWgs84[0], boundsWgs84[2]],
                                              [boundsWgs84[1], boundsWgs84[3]], crs=4326)).to_crs(3857).geometry).total_bounds
bg_map, bg_extent = ctx.bounds2img(bounds[0], bounds[1], bounds[2], bounds[3], zoom=13)

# convert the true bounds of the downloaded map to WGS84 (crs=4326) coordinates
boundsWgsMap = gpd.GeoSeries(gpd.GeoDataFrame(geometry=gpd.points_from_xy([bg_extent[0], bg_extent[1]],
                                              [bg_extent[2], bg_extent[3]], crs=3857)).to_crs(4326).geometry).total_bounds

# display the background map in napari
viewer = napari.Viewer()
viewer.camera.orientation2d=('up','right')
viewer.floating_axes.visible=True
viewer.dims.axis_labels=('lat','lon')
viewer.window.add_plugin_dock_widget('napari', 'Features table widget')

# add the downloaded background map as an image layer, with the correct translation and scale to match the lat/lon coordinates
viewer.add_image(bg_map[:,:,:3][::-1], name='background', opacity=0.9, rgb=True,
                   translate=(boundsWgsMap[1], boundsWgsMap[0]),
                   scale=((boundsWgsMap[3]-boundsWgsMap[1])/bg_map.shape[0], (boundsWgsMap[2]-boundsWgsMap[0])/bg_map.shape[1])

                  )

# add the points of interest as a points layer, using some of the features for coloring
points_layer = viewer.add_points(
    data=df[['lat','lon']].to_numpy(),
    features=df,
    border_color='nature',
    border_color_cycle=['blue', 'green'],
    border_width=0.4,
    face_color='stars',
    face_colormap='reds',
    size=0.002, name='POI'
)

if __name__ == '__main__':
    napari.run()

