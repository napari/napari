"""Utility script to generate copies of icons with colors based
on our themes. Neccessary workaround because qt does not allow
for styling svg elements using qss

run as python -m napari.resources.build_icons"""

from os.path import join

from ..resources import resources_dir
from ..util.theme import palettes

insert = """<style type="text/css">
    path{fill:{{ color }}}
    polygon{fill:{{ color }}}
    circle{fill:{{ color }}}
    rect{fill:{{ color }}}
</style>"""

icons = [
    '2D',
    '3D',
    'add',
    'console',
    'delete',
    'delete_shape',
    'direct',
    'drop_down',
    'ellipse',
    'fill',
    'grid',
    'home',
    'left_arrow',
    'line',
    'minus',
    'move_back',
    'move_front',
    'new_image',
    'new_labels',
    'new_points',
    'new_shapes',
    'new_surface',
    'new_vectors',
    'paint',
    'path',
    'picker',
    'plus',
    'polygon',
    'properties_contract',
    'properties_expand',
    'rectangle',
    'right_arrow',
    'roll',
    'select',
    'transpose',
    'vertex_insert',
    'vertex_remove',
    'visibility',
    'visibility_off',
    'zoom',
]

for name, palette in palettes.items():
    for icon in icons:
        file = icon + '.svg'
        if icon == 'visibility':
            css = insert.replace('{{ color }}', palette['text'])
        elif icon in ['visibility_off', 'menu']:
            css = insert.replace('{{ color }}', palette['highlight'])
        elif icon in [
            'drop_down',
            'plus',
            'minus',
            'properties_contract',
            'properties_expand',
        ]:
            css = insert.replace('{{ color }}', palette['secondary'])
        else:
            css = insert.replace('{{ color }}', palette['icon'])
        with open(join(resources_dir, 'icons', 'svg', file), 'r') as fr:
            contents = fr.readlines()
            fr.close()
            contents.insert(4, css)
            with open(join(resources_dir, 'icons', name, file), 'w') as fw:
                fw.write("".join(contents))
                fw.close()
