import sys
from inspect import Parameter, getdoc, signature

from napari.utils.misc import camel_to_snake
from napari.utils.translations import trans

template = """def {name}{signature}:
    kwargs = locals()
    kwargs.pop('self', None)
    pos_kwargs = dict()
    for name in getattr({cls_name}.__init__, "_deprecated_constructor_args", []):
        pos_kwargs[name] = kwargs.pop(name, None)
    layer = {cls_name}(**kwargs)
    for name, value in pos_kwargs.items():
        if value is not None:
            setattr(layer, name, value)
    self.layers.append(layer)
    return layer
"""


def create_func(cls, name=None, doc=None, filename: str = '<string>'):
    """
    Creates a function (such as `add_<layer>`) to add a layer to the viewer

    The functionality is inherited from the corresponding `<layer>` class.
    """
    cls_name = cls.__name__

    if name is None:
        name = camel_to_snake(cls_name)

    if 'layer' in name:
        raise ValueError(
            trans._(
                "name {name} should not include 'layer'",
                deferred=True,
                name=name,
            )
        )

    name = 'add_' + name

    if doc is None:
        # While the original class may have Attributes in its docstring, the
        # generated function should not have an Attributes section.
        # See https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes
        doc = getdoc(cls)
        start = doc.find('\n\nParameters\n----------\n')
        end = doc.find('\n\nAttributes\n----------\n')
        if end == -1:
            end = None
        if start > 0:
            doc = doc[start:end]

        n = 'n' if cls_name[0].lower() in 'aeiou' else ''
        doc = f'Add a{n} {cls_name} layer to the layer list. ' + doc
        doc += '\n\nReturns\n-------\n'
        doc += f'layer : :class:`napari.layers.{cls_name}`'
        doc += f'\n\tThe newly-created {cls_name.lower()} layer.'
        doc = doc.expandtabs(4)

    sig = signature(cls)
    additional_parameters = []
    if hasattr(cls.__init__, "_deprecated_constructor_args"):
        additional_parameters = [
            Parameter(
                name=arg,
                kind=Parameter.KEYWORD_ONLY,
                default=None,
            )
            for arg in cls.__init__._deprecated_constructor_args
        ]
    new_sig = sig.replace(
        parameters=[
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            *list(sig.parameters.values()),
            *additional_parameters,
        ],
        return_annotation=cls,
    )
    src = template.format(
        name=name,
        signature=new_sig,
        cls_name=cls_name,
    )

    execdict = {cls_name: cls, 'napari': sys.modules.get('napari')}
    code = compile(src, filename=filename, mode='exec')
    exec(code, execdict)
    func = execdict[name]

    func.__doc__ = doc
    func.__signature__ = sig.replace(
        parameters=[
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            *list(sig.parameters.values()),
        ],
        return_annotation=cls,
    )

    return func
