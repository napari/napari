import ast
import inspect
from dataclasses import dataclass
from typing import List, Optional, Type

import numpy as np

import napari
from napari import layers
from napari.components.viewer_model import ViewerModel
from napari.utils.events import EventedModel
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


HEADER = [
    'Class',
    'Event Name',
    'Access At',
    'Emitted when __ changes',
    'Event Attribute(s)',
]
ROW = '| {:13.13} | {:16.16} | {:37.37} | {:40.40} | {:37.37} |'


def iter_evented_model_events(module=napari):
    for mod in walk_modules(module):
        for kls in iter_classes(mod):
            if not issubclass(kls, EventedModel):
                continue
            for name, field_ in kls.__fields__.items():
                finfo = field_.field_info
                if finfo.allow_mutation:
                    descr = f"{finfo.title.lower()}" if finfo.title else ''
                    yield Ev(name, kls, descr, field_.type_)


class BaseEmitterVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._emitters: List[str] = []

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', None) == 'EmitterGroup':
            self._emitters.extend([name.arg for name in node.keywords])


def base_event_names():
    root = ast.parse(Path('napari/layers/base/base.py').read_text())
    visitor = BaseEmitterVisitor()
    visitor.visit(root)
    return visitor._emitters


ex_layers = [
    layers.Image(np.random.random((2, 2))),
    layers.Labels(np.random.randint(20, size=(10, 15))),
    layers.Points(10 * np.random.random((10, 2))),
    layers.Vectors(20 * np.random.random((10, 2, 2))),
    layers.Shapes(20 * np.random.random((10, 4, 2))),
    layers.Surface(
        (
            20 * np.random.random((10, 3)),
            np.random.randint(10, size=(6, 3)),
            np.random.random(10),
        )
    ),
    layers.Tracks(
        np.column_stack(
            (np.ones(20), np.arange(20), 20 * np.random.random((20, 2)))
        )
    ),
]


def iter_layer_events():
    basenames = base_event_names()
    for name in basenames:
        yield Ev(name, layers.Layer)
    for lay in ex_layers:
        for name in [i for i in lay.events.emitters if i not in basenames]:
            yield Ev(name, lay.__class__)


if __name__ == '__main__':
    import re
    from pathlib import Path

    docs = Path(__file__).parent.parent / 'docs'
    file = docs / 'guides' / 'events_reference.md'

    # Do viewer events

    rows = [
        [
            f'`{ev.model.__name__}`',
            f'`{ev.name}`',
            f'`{ev.access_at()}`',
            ev.description or '',
            f'value: `{ev.type_name()}`',
        ]
        for ev in iter_evented_model_events()
        if ev.access_at()
    ]

    text = file.read_text()
    table = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    text = re.sub(
        '(VIEWER EVENTS TABLE -->)([^<!]*)(<!--)', f'\\1\n{table}\n\\3', text
    )
    file.write_text(text)

    # Do layer events
    HEADER.remove('Access At')
    rows = [
        [
            f'`{ev.model.__name__}`',
            f'`{ev.name}`',
            ev.description or '',
            '',
        ]
        for ev in iter_layer_events()
    ]
    print(rows)
    text = file.read_text()
    table = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    text = re.sub(
        '(LAYER EVENTS TABLE -->)([^<!]*)(<!--)', f'\\1\n{table}\n\\3', text
    )
    print(table)
    file.write_text(text)
