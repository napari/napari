"""
Example of making calls to the viewer from a separate script or process
using a ZMQ server started with the --remote-port flag.

Run this example after having started napari on localhost with:
   napari --remote-port 7341
"""

from napari.remote import Client

client = Client()
print(
    f"Started client to napari remote server on {client.host} with port {client.port}"
)

version = client.request('version')
print(f"napari version: {version}")

print("Adding image to napari")
client.request(
    'add_image',
    path=r'https://github.com/napari/napari/raw/master/napari/resources/logo.png',
)

print("Getting data from layer 0")
data = client.request('get_data', layer=0)
