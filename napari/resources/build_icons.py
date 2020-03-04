"""Utility script to generate copies of icons with colors based
on our themes. Neccessary workaround because qt does not allow
for styling svg elements using qss

run as python -m napari.resources.build_icons"""

from os import listdir, makedirs
from os.path import join

from ..resources import resources_dir
from ..utils.theme import palettes

insert = """<style type="text/css">
    path{fill:{{ color }}}
    polygon{fill:{{ color }}}
    circle{fill:{{ color }}}
    rect{fill:{{ color }}}
</style>"""

svgpath = join(resources_dir, 'icons', 'svg')
qrcpath = join(resources_dir, 'res.qrc')
icons = [i.replace('.svg', '') for i in sorted(listdir(svgpath))]

TEXT_ICONS = ['visibility']

HIGHLIGHT_ICONS = ['visibility_off', 'menu']

SECONDARY_ICONS = [
    'drop_down',
    'plus',
    'minus',
    'properties_contract',
    'properties_expand',
]


def build_icons():

    qrc_string = '''
    <!DOCTYPE RCC>
    <RCC version="1.0">
    <qresource>
        <file>icons/cursor/cursor_disabled.png</file>
        <file>icons/cursor/cursor_square.png</file>'''

    for name, palette in palettes.items():
        palette_dir = join(resources_dir, 'icons', name)
        makedirs(palette_dir, exist_ok=True)
        for icon in icons:
            file = icon + '.svg'
            qrc_string += f'\n    <file>icons/{name}/{file}</file>'
            if icon in TEXT_ICONS:
                css = insert.replace('{{ color }}', palette['text'])
            elif icon in HIGHLIGHT_ICONS:
                css = insert.replace('{{ color }}', palette['highlight'])
            elif icon in SECONDARY_ICONS:
                css = insert.replace('{{ color }}', palette['secondary'])
            else:
                css = insert.replace('{{ color }}', palette['icon'])
            with open(join(svgpath, file), 'r') as fr:
                contents = fr.readlines()
                fr.close()
                contents.insert(4, css)
                with open(join(palette_dir, file), 'w') as fw:
                    fw.write("".join(contents))
                    fw.close()

    qrc_string += '''
    </qresource>
    </RCC>
    '''

    with open(qrcpath, 'w') as f:
        f.write(qrc_string)


if __name__ == "__main__":
    from subprocess import run

    qtpy_file = join(resources_dir, 'qt.py')

    build_icons()
    try:
        run(['pyside2-rcc', '-o', qtpy_file, qrcpath])
    except FileNotFoundError:
        run(['pyrcc5', '-o', qtpy_file, qrcpath])

    with open(qtpy_file, "rt") as fin:
        data = fin.read()
        data = data.replace('PySide2', 'qtpy').replace('PyQt5', 'qtpy')
    with open(qtpy_file, "wt") as fin:
        fin.write(data)
