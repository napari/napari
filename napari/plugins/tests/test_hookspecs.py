from napari.plugins import hookspecs
import inspect
from numpydoc.docscrape import FunctionDoc


def _yield_hookspecs(module_name, hookspec_flag='napari_spec'):
    for name, func in vars(module_name).items():
        if hasattr(func, hookspec_flag):
            yield (name, func)


def test_hookspec_naming():
    """All hookspecs should begin with napari_ (until we decide otherwise)"""
    for name, func in _yield_hookspecs(hookspecs):
        assert name.startswith('napari_'), (
            "hookspec '%s' does not start with 'napari_'" % name
        )


def test_docstring_on_hookspec():
    """All hookspecs should have documentation"""
    for name, func in _yield_hookspecs(hookspecs):
        assert func.__doc__, "no docstring for '%s'" % name


def test_annotation_on_hookspec():
    """All hookspecs should have type annotations for all parameters.

    (Use typing.Any to bail out). If the hookspec accepts no parameters,
    then it should declare a return type annotation.  (until we identify a case
    where a hookspec needs to both take no parameters and return nothing)
    """
    for name, func in _yield_hookspecs(hookspecs):
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


def test_docs_match_signature():
    for name, func in _yield_hookspecs(hookspecs):
        sig = inspect.signature(func)
        docs = FunctionDoc(func)
        sig_params = set(sig.parameters)
        doc_params = {p.name for p in docs.get('Parameters')}
        assert sig_params == doc_params, (
            f"Signature parameters for hookspec '{name}' do"
            " not match the parameters listed in the docstring:\n"
            f"{sig_params} != {doc_params}"
        )
