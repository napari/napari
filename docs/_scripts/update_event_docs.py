import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Type

import numpy as np
from _table_maker import table_repr
from numpydoc.docscrape import ClassDoc, Parameter

import napari
from napari import layers
from napari.components.layerlist import LayerList
from napari.components.viewer_model import ViewerModel
from napari.utils.events import EventedModel

DOCS = Path(__file__).parent.parent


@dataclass
class Ev:
    name: str
    model: Type
    description: Optional[str] = None
    type_: Optional[Type] = None

    def access_at(self):
        """Where this event can be accessed (in code)"""
        if issubclass(self.model, layers.Layer):
            return f'layer.events.{self.name}'

        if issubclass(self.model, LayerList):
            if self.name.startswith('selection.'):
                return f'layers.selection.events.{self.name[10:]}'
            return f'layers.events.{self.name}'

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

    def ev_model_row(self) -> List[str]:
        return [
            f'`{self.model.__name__}`',
            f'`{self.name}`',
            f'`{self.access_at()}`',
            self.description or '',
            f'`{self.type_name()}`',
        ]

    def layer_row(self) -> List[str]:
        return [
            f'`{self.model.__name__}`',
            f'`{self.name}`',
            f'`{self.access_at()}`',
            self.description or '',
            '',
        ]


def walk_modules(
    module: ModuleType, pkg='napari', _walked=None
) -> Iterator[ModuleType]:
    """walk all modules in pkg, starting with `module`."""
    if not _walked:
        _walked = set()
    yield module
    _walked.add(module)
    for name in dir(module):
        attr = getattr(module, name)
        if (
            inspect.ismodule(attr)
            and attr.__package__.startswith(pkg)  # type: ignore
            and attr not in _walked
        ):
            yield from walk_modules(attr, pkg, _walked=_walked)


def iter_classes(module: ModuleType) -> Iterator[Type]:
    """iter all classes in module"""
    for name in dir(module):
        attr = getattr(module, name)
        if inspect.isclass(attr) and attr.__module__ == module.__name__:
            yield attr


def class_doc_attrs(kls: Type) -> Dict[str, Parameter]:
    docs = {p.name: " ".join(p.desc) for p in ClassDoc(kls).get('Attributes')}
    docs.update(
        {p.name: " ".join(p.desc) for p in ClassDoc(kls).get('Parameters')}
    )
    return docs


def iter_evented_model_events(module: ModuleType = napari) -> Iterator[Ev]:
    for mod in walk_modules(module):
        for kls in iter_classes(mod):
            if not issubclass(kls, EventedModel):
                continue
            docs = class_doc_attrs(kls)
            for name, field_ in kls.__fields__.items():
                finfo = field_.field_info
                if finfo.allow_mutation:
                    descr = (
                        f"{finfo.title.lower()}"
                        if finfo.title
                        else docs.get(name)
                    )
                    yield Ev(name, kls, descr, field_.type_)


def iter_evented_container_events(
    module: ModuleType = napari, container_class=LayerList
) -> Iterator[Ev]:
    for mod in walk_modules(module):
        for kls in iter_classes(mod):
            if not issubclass(kls, container_class):
                continue
            docs = class_doc_attrs(kls)
            kls_instance = kls()
            for name, emitter in kls_instance.events._emitters.items():
                descr = docs.get(name)
                yield Ev(name, kls, descr, type_=None)
            if hasattr(kls_instance, 'selection'):
                selection = kls_instance.selection
                for name, emitter in selection.events._emitters.items():
                    if name.startswith('_'):
                        # skip private emitters
                        continue
                    name = 'selection.' + name
                    descr = docs.get(name)
                    yield Ev(name, kls, descr, type_=None)


class BaseEmitterVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._emitters: List[str] = []

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', None) == 'EmitterGroup':
            self._emitters.extend([name.arg for name in node.keywords])  # type: ignore


def base_event_names() -> List[str]:
    from napari.layers.base import base

    root = ast.parse(Path(base.__file__).read_text())
    visitor = BaseEmitterVisitor()
    visitor.visit(root)
    return visitor._emitters


def iter_layer_events() -> Iterator[Ev]:
    basenames = base_event_names()
    docs = class_doc_attrs(layers.Layer)
    for name in basenames:
        yield Ev(name, layers.Layer, description=docs.get(name))

    EXAMPLE_LAYERS: List[layers.Layer] = [
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

    for lay in EXAMPLE_LAYERS:
        docs = class_doc_attrs(type(lay))
        for name in [i for i in lay.events.emitters if i not in basenames]:
            yield Ev(name, lay.__class__, description=docs.get(name))


def merge_image_and_label_rows(rows: List[List[str]]):
    """Merge events common to _ImageBase or IntensityVisualizationMixin."""
    # find events that are common across both Image, Labels and Surface layers.
    image_events = {r[1] for r in rows if r[0] == '`Image`'}
    labels_events = {r[1] for r in rows if r[0] == '`Labels`'}
    surface_events = {r[1] for r in rows if r[0] == '`Surface`'}
    common_events = image_events & labels_events & surface_events
    # common only to Image and Labels
    imagebase_events = (image_events & labels_events) - common_events

    # drop duplicate Labels and/or Surface entries
    rows = [
        r
        for r in rows
        if not (r[0] in ['`Labels`', '`Surface`'] and r[1] in common_events)
    ]
    rows = [
        r
        for r in rows
        if not (r[0] in ['`Labels`', '`Surface`'] and r[1] in imagebase_events)
    ]

    # modify the class name of the Image entries to mention Labels, Surface
    rows = [
        ['`Image`, `Labels`'] + r[1:]
        if r[0] == '`Image`' and r[1] in imagebase_events
        else r
        for r in rows
    ]
    rows = [
        ['`Image`, `Labels`, `Surface`'] + r[1:]
        if r[0] == '`Image`' and r[1] in common_events
        else r
        for r in rows
    ]
    return rows


def main():
    HEADER = [
        'Event',
        'Description',
        'Event.value type',
    ]

    # Do viewer events
    rows = [
        ev.ev_model_row()[2:]
        for ev in iter_evented_model_events()
        if ev.access_at()
    ]
    table1 = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    (DOCS / 'guides' / '_viewer_events.md').write_text(table1)

    # Do LayerList events
    rows = [
        ev.layer_row()[2:]
        for ev in iter_evented_container_events(
            napari, container_class=LayerList
        )
        if ev.access_at()
    ]
    table2 = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    (DOCS / 'guides' / '_layerlist_events.md').write_text(table2)

    # Do layer events
    HEADER = [
        'Class',
        'Event',
        'Description',
        'Event.value type',
    ]
    rows = [
        [ev.layer_row()[0]] + ev.layer_row()[2:] for ev in iter_layer_events()
    ]
    rows = merge_image_and_label_rows(rows)
    table3 = table_repr(rows, padding=2, header=HEADER, divide_rows=False)
    (DOCS / 'guides' / '_layer_events.md').write_text(table3)


if __name__ == '__main__':
    main()
