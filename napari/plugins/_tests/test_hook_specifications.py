import inspect

import pytest
from numpydoc.docscrape import FunctionDoc

from napari.plugins import hook_specifications

# 1. we first create a hook specification decorator:
#    ``napari_hook_specification = napari_plugin_engine.HookSpecificationMarker("napari")``
# 2. when it decorates a function, that function object gets a new attribute
#    called "napari_spec"
# 3. that attribute is what makes specifications discoverable when you run
#    ``plugin_manager.add_hookspecs(module)``
#    (The ``add_hookspecs`` method basically just looks through the module for
#    any functions that have a "napari_spec" attribute.
#
# here, we are using that attribute to discover all of our internal hook
# specifications (in module ``napari.plugins.hook_specifications``) so as to
# make sure that they conform to our own internal rules about documentation and
# type annotations, etc...


HOOK_SPECIFICATIONS = [
    (name, func)
    for name, func in vars(hook_specifications).items()
    if hasattr(func, 'napari_spec')
]


@pytest.mark.parametrize("name, func", HOOK_SPECIFICATIONS)
def test_hook_specification_naming(name, func):
    """All hook specifications should begin with napari_."""
    assert name.startswith('napari_'), (
        "hook specification '%s' does not start with 'napari_'" % name
    )


@pytest.mark.parametrize("name, func", HOOK_SPECIFICATIONS)
def test_docstring_on_hook_specification(name, func):
    """All hook specifications should have documentation."""
    assert func.__doc__, "no docstring for '%s'" % name


@pytest.mark.parametrize("name, func", HOOK_SPECIFICATIONS)
def test_annotation_on_hook_specification(name, func):
    """All hook specifications should have type annotations for all parameters.

    (Use typing.Any to bail out). If the hook specification accepts no
    parameters, then it should declare a return type annotation.  (until we
    identify a case where a hook specification needs to both take no parameters
    and return nothing)
    """
    sig = inspect.signature(func)
    if sig.parameters:
        for param in sig.parameters.values():
            for forbidden in ('_plugin', '_skip_impls', '_return_result_obj'):
                assert (
                    param.name != forbidden
                ), f'Must not name hook_specification argument "{forbidden}".'
            assert param.annotation is not param.empty, (
                f"in hook specification '{name}', parameter '{param}' "
                "has no type annotation"
            )
    else:
        assert sig.return_annotation is not sig.empty, (
            f"hook specifications with no parameters ({name}),"
            " must declare a return type annotation"
        )


@pytest.mark.parametrize("name, func", HOOK_SPECIFICATIONS)
def test_docs_match_signature(name, func):
    sig = inspect.signature(func)
    docs = FunctionDoc(func)
    sig_params = set(sig.parameters)
    doc_params = {p.name for p in docs.get('Parameters')}
    assert sig_params == doc_params, (
        f"Signature parameters for hook specification '{name}' do "
        "not match the parameters listed in the docstring:\n"
        f"{sig_params} != {doc_params}"
    )
