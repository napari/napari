import ast
import inspect
from dataclasses import dataclass
from typing import List, Optional, Type

import napari
from napari import layers
from napari.components.viewer_model import ViewerModel
from napari.utils.events import EventedModel
from napari.utils.settings._defaults import BaseNapariSettings
from tools._table_maker import table_repr


def walk_modules(module, pkg='napari', _walked=None):
    if not _walked:
        _walked = set()
    yield module
    _walked.add(module)
    for name in dir(module):
        attr = getattr(module, name)
        if (
            inspect.ismodule(attr)
            and attr.__package__.startswith(pkg)
            and attr not in _walked
        ):
            yield from walk_modules(attr, pkg, _walked=_walked)


def iter_classes(module):
    for name in dir(module):
        attr = getattr(module, name)
        if inspect.isclass(attr) and attr.__module__ == module.__name__:
            yield attr


@dataclass
class Ev:
    name: str
    model: Type
    description: str = ''
    type_: Optional[Type] = None

    def access_at(self):
        if issubclass(self.model, layers.Layer):
            return f'layer.events.{self.name}'

        if issubclass(self.model, ViewerModel):
            return f'viewer.events.{self.name}'
        for name, field_ in napari.Viewer.__fields__.items():
            if field_.type_ is self.model:
                return f'viewer.{name}.events.{self.name}'
        return ''

    def type_name(self):
        if cls_name := getattr(self.type_, '__name__', None):
            return cls_name
        name = str(self.type_) if self.type_ else ''
        return name.replace("typing.", "")


HEADER = (
    'Class',
    'Event Name',
    'Access At',
    'Emitted when __ changes',
    'Event Attribute(s)',
)
ROW = '| {:13.13} | {:16.16} | {:37.37} | {:40.40} | {:37.37} |'


def iter_evented_model_events(module=napari):
    for mod in walk_modules(module):
        for kls in iter_classes(mod):
            if not issubclass(kls, EventedModel) or issubclass(
                kls, BaseNapariSettings
            ):
                continue
            for name, field_ in kls.__fields__.items():
                finfo = field_.field_info
                if finfo.allow_mutation:
                    descr = f"{finfo.title.lower()}" if finfo.title else ''
                    yield Ev(name, kls, descr, field_.type_)


# class FindEmitterVisitor(ast.NodeVisitor):
#     def __init__(self, model) -> None:
#         self._model = model
#         super().__init__()
#         self._emitters: List[Ev] = []

#     def visit_Call(self, node: ast.Call):
#         if (
#             getattr(node.func, 'attr', None) == 'add'
#             and getattr(node.func.value, 'attr', None) == 'events'
#         ):
#             for name in node.keywords:
#                 self._emitters.append(Ev(name.arg, self._model))


# def iter_layer_events():

#     for name in ['Layer'] + sorted(layers.NAMES):
#         n = 'base' if name == 'Layer' else name.lower()
#         file = f'napari/layers/{n}/{n}.py'
#         root = ast.parse(Path(file).read_text())
#         for node in ast.walk(root):
#             for child in ast.iter_child_nodes(node):
#                 child.parent = node

#         visitor = FindEmitterVisitor(getattr(layers, name.title()))
#         visitor.visit(root)
#         yield from visitor._emitters


def iter_layer_events():
    import numpy as np

    for name in sorted(layers.NAMES):
        cls = getattr(layers, name.title())
        try:
            lay = cls(np.zeros((2, 2), dtype='uint8'))
        except:
            lay = cls([np.zeros((3, 4), dtype='uint8')] * 3)


if __name__ == '__main__':
    import re
    from pathlib import Path

    docs = Path(__file__).parent.parent / 'docs'
    file = docs / 'guides' / 'events_reference.md'

    # Do viewer events

    # rows = [
    #     [
    #         f'`{ev.model.__name__}`',
    #         f'`{ev.name}`',
    #         f'`{ev.access_at()}`',
    #         ev.description or '',
    #         f'value: `{ev.type_name()}`',
    #     ]
    #     for ev in iter_evented_model_events()
    #     if ev.access_at()
    # ]

    # text = file.read_text()
    # table = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    # text = re.sub(
    #     '(VIEWER EVENTS TABLE -->)([^<!]*)(<!--)', f'\\1\n{table}\n\\3', text
    # )
    # file.write_text(text)

    # Do layer events

    iter_layer_events()
    # rows = [
    #     [
    #         f'`{ev.model.__name__}`',
    #         f'`{ev.name}`',
    #         ev.access_at(),
    #         ev.description or '',
    #         f'',
    #     ]
    #     for ev in iter_layer_events()
    # ]
    # print(rows)
    # text = file.read_text()
    # table = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    # text = re.sub(
    #     '(LAYER EVENTS TABLE -->)([^<!]*)(<!--)', f'\\1\n{table}\n\\3', text
    # )
    # print(table)
    # # file.write_text(text)
