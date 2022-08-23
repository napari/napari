from functools import partial


def _load_skimage_data(name, **kwargs):
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
    elif name == 'binary_blobs_3D':
        kwargs['n_dim'] = 3
        kwargs.setdefault('length', 128)
        kwargs.setdefault('volume_fraction', 0.25)
        name = 'binary_blobs'

    return [(getattr(skimage.data, name)(**kwargs), {'name': name})]


# fmt: off
SKIMAGE_DATA = [
    'astronaut', 'binary_blobs', 'binary_blobs_3D', 'brain', 'brick', 'camera', 'cat',
    'cell', 'cells3d', 'checkerboard', 'clock', 'coffee', 'coins', 'colorwheel',
    'eagle', 'grass', 'gravel', 'horse', 'hubble_deep_field', 'human_mitosis',
    'immunohistochemistry', 'kidney', 'lfw_subset', 'lily', 'microaneurysms', 'moon',
    'page', 'retina', 'rocket', 'shepp_logan_phantom', 'skin', 'text',
]

globals().update({key: partial(_load_skimage_data, key) for key in SKIMAGE_DATA})
