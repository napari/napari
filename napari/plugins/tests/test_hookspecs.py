import inspect
import warnings

import pytest
from numpydoc.docscrape import FunctionDoc

from napari.plugins import hookspecs

# 1. we first create a hookspec decorator:
#    ``hookspec = pluggy.HookspecMarker("napari")```
# 2. when it decorates a function, that function object gets a new attribute
#    called "napari_spec"
# 3. that attribute is what makes them discoverable when you run
#    ```plugin_manager.add_hookspecs(module)```
#
# here, we are using that attribute to discover all of our internal hookspecs
# (in module ``napari.plugins.hookspecs``) so as to make sure that they conform
# to our own internal rules about documentation and type annotations, etc...

HOOKSPECS = []
for name, func in vars(hookspecs).items():
    if hasattr(func, 'napari_spec'):
        HOOKSPECS.append((name, func))


@pytest.mark.parametrize("name, func", HOOKSPECS)
def test_hookspec_naming(name, func):
    """All hookspecs should begin with napari_ (until we decide otherwise)"""
    assert name.startswith('napari_'), (
        "hookspec '%s' does not start with 'napari_'" % name
    )


@pytest.mark.parametrize("name, func", HOOKSPECS)
def test_docstring_on_hookspec(name, func):
    """All hookspecs should have documentation"""
    assert func.__doc__, "no docstring for '%s'" % name


@pytest.mark.parametrize("name, func", HOOKSPECS)
def test_annotation_on_hookspec(name, func):
    """All hookspecs should have type annotations for all parameters.

    (Use typing.Any to bail out). If the hookspec accepts no parameters,
    then it should declare a return type annotation.  (until we identify a case
    where a hookspec needs to both take no parameters and return nothing)
    """
    sig = inspect.signature(func)
    if sig.parameters:
        for param in sig.parameters.values():
            assert param.annotation is not param.empty, (
                f"in hookspec '{name}', parameter '{param}' "
                "has no type annotation"
            )
    else:
        assert sig.return_annotation is not sig.empty, (
            f"hookspecs with no parameters ({name}),"
            " must declare a return type annotation"
        )


@pytest.mark.parametrize("name, func", HOOKSPECS)
def test_docs_match_signature(name, func):
    sig = inspect.signature(func)
    docs = FunctionDoc(func)
    sig_params = set(sig.parameters)
    doc_params = {p.name for p in docs.get('Parameters')}
    assert sig_params == doc_params, (
        f"Signature parameters for hookspec '{name}' do "
        "not match the parameters listed in the docstring:\n"
        f"{sig_params} != {doc_params}"
    )

    # we know the parameters names match, now check that their types match...
    # but only emit a warning if not
    for doc_param in docs.get('Parameters'):
        sig_param = sig.parameters.get(doc_param.name)
        name = getattr(sig_param.annotation, '_name', None)
        name = name or getattr(sig_param.annotation, '__name__', None)
        if doc_param.type != name:
            warnings.warn(
                SyntaxWarning(
                    f'The type ({name}) for parameter '
                    f'"{sig_param.name}" in hookspec "{name}" does not '
                    'match the type specified in the docstring '
                    f'({doc_param.type})'
                )
            )
