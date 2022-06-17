from functools import partial

from ..utils.translations import trans

SKIMAGE_DATA = [
    ('astronaut', trans._('Astronaut (RGB)')),
    ('binary_blobs', trans._('Binary Blobs')),
    ('binary_blobs_3D', trans._('Binary Blobs (3D)')),
    ('brain', trans._('Brain (3D)')),
    ('brick', trans._('Brick')),
    ('camera', trans._('Camera')),
    ('cat', trans._('Cat (RGB)')),
    ('cell', trans._('Cell')),
    ('cells3d', trans._('Cells (3D+2Ch)')),
    ('checkerboard', trans._('Checkerboard')),
    ('clock', trans._('Clock')),
    ('coffee', trans._('Coffee (RGB)')),
    ('coins', trans._('Coins')),
    ('colorwheel', trans._('Colorwheel (RGB)')),
    ('eagle', trans._('Eagle')),
    ('grass', trans._('Grass')),
    ('gravel', trans._('Gravel')),
    ('horse', trans._('Horse')),
    ('hubble_deep_field', trans._('Hubble Deep Field (RGB)')),
    ('human_mitosis', trans._('Human Mitosis')),
    ('immunohistochemistry', trans._('Immunohistochemistry (RGB)')),
    ('kidney', trans._('Kidney (3D+3Ch)')),
    ('lfw_subset', trans._('Labeled Faces in the Wild')),
    ('lily', trans._('Lily (4Ch)')),
    ('microaneurysms', trans._('Microaneurysms')),
    ('moon', trans._('Moon')),
    ('page', trans._('Page')),
    ('retina', trans._('Retina (RGB)')),
    ('rocket', trans._('Rocket (RGB)')),
    ('shepp_logan_phantom', trans._('Shepp Logan Phantom')),
    ('skin', trans._('Skin (RGB)')),
    ('text', trans._('Text')),
]


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


_DATA = {
    key: {'data': partial(_load_skimage_data, key), 'display_name': dname}
    for (key, dname) in SKIMAGE_DATA
}


globals().update({k: v['data'] for k, v in _DATA.items()})
