"""Utility script to generate copies of icons with colors based
on our themes. Neccessary workaround because qt does not allow
for styling svg elements using qss

run as python -m napari.resources.build_icons"""

from os import listdir, makedirs
from os.path import exists, join
from subprocess import run

from ..resources import resources_dir
from ..utils.theme import palettes

insert = """<style type="text/css">
    path{fill:{{ color }}}
    polygon{fill:{{ color }}}
    circle{fill:{{ color }}}
    rect{fill:{{ color }}}
</style>"""

SVGPATH = join(resources_dir, 'icons', 'svg')
QRCPATH = join(resources_dir, 'res.qrc')
ICONS = [i.replace('.svg', '') for i in sorted(listdir(SVGPATH))]

TEXT_ICONS = ['visibility']

HIGHLIGHT_ICONS = ['visibility_off', 'menu']

SECONDARY_ICONS = [
    'drop_down',
    'plus',
    'minus',
    'properties_contract',
    'properties_expand',
]


def build_resources(qrcpath=None, overwrite=False):
    qrcpath = qrcpath or QRCPATH

    if exists(qrcpath) and (not overwrite):
        return qrcpath

    qrc_string = '''
    <!DOCTYPE RCC>
    <RCC version="1.0">
    <qresource>
        <file>icons/cursor/cursor_disabled.png</file>
        <file>icons/cursor/cursor_square.png</file>'''

    for name, palette in palettes.items():
        palette_dir = join(resources_dir, 'icons', name)
        makedirs(palette_dir, exist_ok=True)
        for icon in ICONS:
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
            with open(join(SVGPATH, file), 'r') as fr:
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

    return qrcpath


def build_python_resources(res_qrc, out_path, overwrite=False):

    if exists(out_path):
        return

    try:
        run(['pyrcc5', '-o', out_path, res_qrc])
    except FileNotFoundError:
        run(['pyside2-rcc', '-o', out_path, res_qrc])

    with open(out_path, "rt") as fin:
        data = fin.read()
        data = data.replace('PySide2', 'qtpy').replace('PyQt5', 'qtpy')
    with open(out_path, "wt") as fin:
        fin.write(data)


def build_icons(out_path=None, overwrite=False):
    out_path = out_path or join(resources_dir, 'qt.py')
    resources_path = build_resources()
    build_python_resources(resources_path, out_path)
    return out_path


if __name__ == "__main__":
    build_icons(overwrite=True)
