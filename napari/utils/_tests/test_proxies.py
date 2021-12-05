import pytest

from napari.utils._proxies import PublicOnlyProxy, ReadOnlyWrapper


def test_ReadOnlyWrapper_setitem():
    """test that ReadOnlyWrapper prevents setting items"""
    d = {'hi': 3}
    d_read_only = ReadOnlyWrapper(d)

    with pytest.raises(TypeError):
        d_read_only['hi'] = 5


def test_ReadOnlyWrapper_setattr():
    """test that ReadOnlyWrapper prevents setting attributes"""

    class TestClass:
        x = 3

    tc = TestClass()
    tc_read_only = ReadOnlyWrapper(tc)

    with pytest.raises(TypeError):
        tc_read_only.x = 5


def test_PublicOnlyProxy():
    class X:
        a = 1
        _b = 'nope'

        def method(self):
            return 2

    class Tester:
        x = X()
        _private = 2

        def __getitem__(self, key):
            return X()

    t = Tester()
    proxy = PublicOnlyProxy(t)
    assert proxy.x.a == 1
    assert proxy[0].a == 1
    assert proxy.x.method() == 2

    assert isinstance(proxy, Tester)
    with pytest.warns(FutureWarning) as e:
        proxy._private
        assert 'Private attribute access' in str(e[0].message)

    with pytest.warns(FutureWarning) as e:
        # works on sub-objects too
        proxy.x._b
        assert 'Private attribute access' in str(e[0].message)

    with pytest.warns(FutureWarning) as e:
        # works on sub-items too
        proxy[0]._b
        assert 'Private attribute access' in str(e[0].message)

    assert '_private' not in dir(proxy)
    assert '_private' in dir(t)
