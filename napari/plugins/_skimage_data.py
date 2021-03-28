from napari_plugin_engine import napari_hook_implementation

# fmt: off
SKIMAGE_DATA = [
    'astronaut', 'binary_blobs', 'brain', 'brick', 'camera', 'cat', 'cell',
    'checkerboard', 'clock', 'coffee', 'coins',
    'colorwheel', 'eagle', 'grass', 'gravel', 'horse', 'hubble_deep_field',
    'human_mitosis', 'immunohistochemistry', 'kidney', 'lfw_subset', 'lily',
    'microaneurysms', 'moon', 'page', 'retina', 'rocket',
    'shepp_logan_phantom', 'skin', 'text'
]
# fmt: on


def _load_skimage_data(name):
    import skimage.data

    if name == 'cells3d':
        return [
            (
                skimage.data.cells3d(),
                {
                    'channel_axis': 1,
                    'name': ['membrane', 'nuclei'],
                    'contrast_limits': [(1110, 23855), (1600, 50000)],
                },
            )
        ]
    elif name == 'kidney':
        return [
            (
                skimage.data.kidney(),
                {
                    'channel_axis': -1,
                    'name': ['nuclei', 'WGA', 'actin'],
                    'colormap': ['blue', 'green', 'red'],
                },
            )
        ]
    elif name == 'lily':
        return [
            (
                skimage.data.lily(),
                {
                    'channel_axis': -1,
                    'name': ['lily-R', 'lily-G', 'lily-W', 'lily-B'],
                    'colormap': ['red', 'green', 'gray', 'blue'],
                },
            )
        ]

    return [(getattr(skimage.data, name)(), {'name': name})]


@napari_hook_implementation
def napari_provide_sample_data():
    from functools import partial

    return {n: partial(_load_skimage_data, n) for n in SKIMAGE_DATA}
