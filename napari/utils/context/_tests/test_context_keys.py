import pytest

from napari.utils.context._context_keys import (
    MISSING,
    ContextKey,
    ContextKeyInfo,
    ContextNamespace,
)


def test_context_key_info():
    info = ContextKey.info()
    assert isinstance(info, list) and len(info)
    assert all(isinstance(x, ContextKeyInfo) for x in info)


def _adder(x: list) -> int:
    return sum(x)


def test_context_namespace():
    class Ns(ContextNamespace):
        my_key = ContextKey[list, int](0, 'description', _adder)
        optional_key = ContextKey[None, str](description='might be missing')

    assert 'my_key' in Ns.__members__
    assert str(Ns.my_key) == 'my_key'

    assert any(x.description == 'description' for x in ContextKey.info())
    # make sure the type hints were inferred from adder
    assert Ns.my_key.__orig_class__.__args__ == (list, int)  # type: ignore

    assert isinstance(Ns.my_key, ContextKey)

    ctx = {}
    ns = Ns(ctx)

    assert ns.my_key == 0
    assert ctx['my_key'] == 0
    ns.my_key = 2
    assert ctx['my_key'] == 2

    assert 'optional_key' not in ctx
    assert ns.optional_key is MISSING
    ns.reset('optional_key')  # shouldn't raise error to reset a missing key
    # maybe the key is there though
    ctx['optional_key'] = 'hi'
    assert ns.optional_key == 'hi'

    ns.reset_all()
    assert ctx['my_key'] == 0
    assert 'optional_key' not in ctx
    assert repr(ns) == "{'my_key': 0, 'optional_key': MISSING}"


def test_good_naming():
    with pytest.raises(RuntimeError):
        # you're not allowed to create a key with an id different from
        # it's attribute name
        class Ns(ContextNamespace):
            my_key = ContextKey(id='not_my_key')  # type: ignore
