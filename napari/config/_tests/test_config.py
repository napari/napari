import os
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager

import pytest
import yaml

from napari import config
from napari.layers.image._image_constants import Interpolation


def test_canonical_name():
    c = {"foo-bar": 1, "fizz_buzz": 2}
    assert config.canonical_name("foo-bar", c) == "foo-bar"
    assert config.canonical_name("foo_bar", c) == "foo-bar"
    assert config.canonical_name("fizz-buzz", c) == "fizz_buzz"
    assert config.canonical_name("fizz_buzz", c) == "fizz_buzz"
    assert config.canonical_name("new-key", c) == "new-key"
    assert config.canonical_name("new_key", c) == "new_key"


def test_update():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": OrderedDict({"b": 2})}
    config.update(b, a)
    assert b == {"x": 1, "y": {"a": 1, "b": 2}, "z": 3}

    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"a": 3, "b": 2}}
    config.update(b, a, priority="old")
    assert b == {"x": 2, "y": {"a": 3, "b": 2}, "z": 3}


def test_merge():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {"x": 2, "y": {"a": 1, "b": 2}, "z": 3}

    conf = config.merge(a, b)
    assert conf == expected


def test_collect_yaml_paths(tmp_path):
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {"x": 2, "y": {"a": 1, "b": 2}, "z": 3}

    fn1 = tmp_path / 'a.yaml'
    fn2 = tmp_path / 'b.yaml'

    with open(fn1, "w") as f:
        yaml.dump(a, f)
    with open(fn2, "w") as f:
        yaml.dump(b, f)

    conf = config.merge(*config.core.collect_yaml(paths=[fn1, fn2]))
    assert conf == expected


def test_collect_yaml_dir(tmp_path):
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {"x": 2, "y": {"a": 1, "b": 2}, "z": 3}

    fn1 = tmp_path / "a.yaml"
    fn2 = tmp_path / "b.yaml"

    with open(fn1, "w") as f:
        yaml.dump(a, f)
    with open(fn2, "w") as f:
        yaml.dump(b, f)

    conf = config.merge(*config.core.collect_yaml(paths=[tmp_path]))
    assert conf == expected


def test_collect_yaml_with_private(tmp_path):
    """collect_yaml should ignore files starting with underscore"""
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    fn1 = tmp_path / "a.yaml"
    fn2 = tmp_path / "_b.yaml"

    with open(fn1, "w") as f:
        yaml.dump(a, f)
    with open(fn2, "w") as f:
        yaml.dump(b, f)

    conf = config.merge(*config.core.collect_yaml(paths=[tmp_path]))
    assert conf == a


@contextmanager
def no_read_permissions(path):
    perm_orig = stat.S_IMODE(os.stat(path).st_mode)
    perm_new = perm_orig ^ stat.S_IREAD
    try:
        os.chmod(path, perm_new)
        yield
    finally:
        os.chmod(path, perm_orig)


# in local linux tests, the no_read_permissions context works, but it fails
# to modify permissions on cirrus CI for some reason.  So skipping linux here.
# it still works on mac CI, which at least confirms the desired behavior.
@pytest.mark.skipif(
    sys.platform in ("win32", "linux"), reason="Can't make writeonly file"
)
@pytest.mark.parametrize("kind", ["directory", "file"])
def test_collect_yaml_permission_errors(tmpdir, kind):
    a = {"x": 1, "y": 2}
    b = {"y": 3, "z": 4}

    dir_path = str(tmpdir)
    a_path = os.path.join(dir_path, "a.yaml")
    b_path = os.path.join(dir_path, "b.yaml")

    with open(a_path, mode="w") as f:
        yaml.dump(a, f)
    with open(b_path, mode="w") as f:
        yaml.dump(b, f)

    if kind == "directory":
        cant_read = dir_path
        expected = {}
    else:
        cant_read = a_path
        expected = b

    with no_read_permissions(cant_read):
        conf = config.merge(*config.core.collect_yaml(paths=[dir_path]))
        assert conf == expected


def test_env():
    env = {
        "NAPARI_A_B": "123",
        "NAPARI_C": "True",
        "NAPARI_D": "hello",
        "NAPARI_E__X": "123",
        "NAPARI_E__Y": "456",
        "NAPARI_F": '[1, 2, "3"]',
        "NAPARI_G": "/not/parsable/as/literal",
        "FOO": "not included",
    }

    expected = {
        "a_b": 123,
        "c": True,
        "d": "hello",
        "e": {"x": 123, "y": 456},
        "f": [1, 2, "3"],
        "g": "/not/parsable/as/literal",
    }

    res = config.core.collect_env(env)
    res.pop("_dirty")
    assert res == expected


def test_collect(tmp_path):
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}
    env = {"NAPARI_W": 4}

    expected = {"w": 4, "x": 2, "y": {"a": 1, "b": 2}, "z": 3}

    fn1 = tmp_path / "fn1.yaml"
    fn2 = tmp_path / "fn2.yaml"

    with open(fn1, "w") as f:
        yaml.dump(a, f)
    with open(fn2, "w") as f:
        yaml.dump(b, f)

    conf = config.collect([fn1, fn2], env=env)
    conf.pop("_dirty", None)
    assert conf == expected


def test_collect_env_none():
    os.environ["NAPARI_FOO"] = "bar"
    try:
        conf = config.collect([])
        conf.pop("_dirty", None)
        assert conf == {"foo": "bar"}
    finally:
        del os.environ["NAPARI_FOO"]


def test_get():
    d = {"x": 1, "y": {"a": 2}}

    assert config.get("x", config=d) == 1
    assert config.get("y.a", config=d) == 2
    assert config.get("y.b", 123, config=d) == 123
    with pytest.raises(KeyError):
        config.get("y.b", config=d)


def test_pop():
    d = {"x": 1, "y": {"a": 2, "b": 3}}

    assert config.pop("y.a", config=d) == 2
    d.pop("_dirty", None)
    assert d == {"x": 1, "y": {"b": 3}}
    assert config.pop("x", config=d) == 1
    d.pop("_dirty", None)
    assert d == {"y": {"b": 3}}
    assert config.pop("y.c", 123, config=d) == 123
    assert config.pop("y", config=d) == {"b": 3}
    d.pop("_dirty", None)
    assert d == {}

    with pytest.raises(KeyError):
        config.pop("z", config=d)


def test_ensure_file(tmpdir):
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 123}

    source = os.path.join(str(tmpdir), "source.yaml")
    dest = os.path.join(str(tmpdir), "dest")
    destination = os.path.join(dest, "source.yaml")

    with open(source, "w") as f:
        yaml.dump(a, f)

    config.ensure_file(source=source, destination=dest, comment=False)

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert result == a

    # don't overwrite old config files
    with open(source, "w") as f:
        yaml.dump(b, f)

    config.ensure_file(source=source, destination=dest, comment=False)

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert result == a

    os.remove(destination)

    # Write again, now with comments
    config.ensure_file(source=source, destination=dest, comment=True)

    with open(destination) as f:
        text = f.read()
    assert "123" in text

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert not result


def test_set():
    with config.set(abc=123):
        assert config.config["abc"] == 123
        with config.set(abc=456):
            assert config.config["abc"] == 456
        assert config.config["abc"] == 123

    assert "abc" not in config.config

    with config.set({"abc": 123}):
        assert config.config["abc"] == 123
    assert "abc" not in config.config

    with config.set({"abc.x": 1, "abc.y": 2, "abc.z.a": 3}):
        assert config.config["abc"] == {"x": 1, "y": 2, "z": {"a": 3}}
    assert "abc" not in config.config

    d = {}
    config.set({"abc.x": 123}, config=d)
    assert d["abc"]["x"] == 123


def test_set_kwargs():
    with config.set(foo__bar=1, foo__baz=2):
        assert config.config["foo"] == {"bar": 1, "baz": 2}
    assert "foo" not in config.config

    # Mix kwargs and dict, kwargs override
    with config.set({"foo.bar": 1, "foo.baz": 2}, foo__buzz=3, foo__bar=4):
        assert config.config["foo"] == {"bar": 4, "baz": 2, "buzz": 3}
    assert "foo" not in config.config

    # Mix kwargs and nested dict, kwargs override
    with config.set({"foo": {"bar": 1, "baz": 2}}, foo__buzz=3, foo__bar=4):
        assert config.config["foo"] == {"bar": 4, "baz": 2, "buzz": 3}
    assert "foo" not in config.config


def test_set_nested():
    with config.set({"abc": {"x": 123}}):
        assert config.config["abc"] == {"x": 123}
        with config.set({"abc.y": 456}):
            assert config.config["abc"] == {"x": 123, "y": 456}
        assert config.config["abc"] == {"x": 123}
    assert "abc" not in config.config


def test_set_hard_to_copyables():
    import threading

    with config.set(x=threading.Lock()):
        with config.set(y=1):
            pass


@pytest.mark.parametrize("mkdir", [True, False])
def test_ensure_file_directory(mkdir, tmpdir):
    a = {"x": 1, "y": {"a": 1}}

    source = os.path.join(str(tmpdir), "source.yaml")
    dest = os.path.join(str(tmpdir), "dest")

    with open(source, "w") as f:
        yaml.dump(a, f)

    if mkdir:
        os.mkdir(dest)

    config.ensure_file(source=source, destination=dest)

    assert os.path.isdir(dest)
    assert os.path.exists(os.path.join(dest, "source.yaml"))


def test_ensure_file_defaults_to_NAPARI_CONFIG_directory(tmpdir):
    a = {"x": 1, "y": {"a": 1}}
    source = os.path.join(str(tmpdir), "source.yaml")
    with open(source, "w") as f:
        yaml.dump(a, f)

    destination = os.path.join(str(tmpdir), "napari")
    PATH = config.core.PATH
    try:
        config.core.PATH = destination
        config.ensure_file(source=source)
    finally:
        config.core.PATH = PATH

    assert os.path.isdir(destination)
    [fn] = os.listdir(destination)
    assert os.path.split(fn)[1] == os.path.split(source)[1]


def test_rename():
    aliases = {"foo_bar": "foo.bar"}
    conf = {"foo-bar": 123}
    config.rename(aliases, config=conf)
    conf.pop("_dirty", None)
    assert conf == {"foo": {"bar": 123}}


def test_refresh():
    defaults = []
    conf = {}

    config.update_defaults({"a": 1}, config=conf, defaults=defaults)
    conf.pop("_dirty", None)
    assert conf == {"a": 1}

    config.refresh(
        paths=[], env={"NAPARI_B": "2"}, config=conf, defaults=defaults
    )
    conf.pop("_dirty", None)
    assert conf == {"a": 1, "b": 2}

    config.refresh(
        paths=[], env={"NAPARI_C": "3"}, config=conf, defaults=defaults
    )
    conf.pop("_dirty", None)
    assert conf == {"a": 1, "c": 3}


@pytest.mark.parametrize(
    "inp,out",
    [
        ("1", "1"),
        (1, 1),
        ("$FOO", "foo"),
        ([1, "$FOO"], [1, "foo"]),
        ((1, "$FOO"), (1, "foo")),
        ({1, "$FOO"}, {1, "foo"}),
        ({"a": "$FOO"}, {"a": "foo"}),
        ({"a": "A", "b": [1, "2", "$FOO"]}, {"a": "A", "b": [1, "2", "foo"]}),
    ],
)
def test_expand_environment_variables(inp, out):
    try:
        os.environ["FOO"] = "foo"
        assert config.utils.expand_environment_variables(inp) == out
    finally:
        del os.environ["FOO"]


def test_env_var_canonical_name(monkeypatch):
    value = 3
    monkeypatch.setenv("NAPARI_A_B", str(value))
    d = {}
    config.refresh(config=d)
    assert config.get("a_b", config=d) == value
    assert config.get("a-b", config=d) == value


def test_get_set_canonical_name():
    c = {"x-y": {"a_b": 123}}

    keys = ["x_y.a_b", "x-y.a-b", "x_y.a-b"]
    for k in keys:
        assert config.get(k, config=c) == 123

    with config.set({"x_y": {"a-b": 456}}, config=c):
        for k in keys:
            assert config.get(k, config=c) == 456

    # No change to new keys in sub dicts
    with config.set({"x_y": {"a-b": {"c_d": 1}, "e-f": 2}}, config=c):
        assert config.get("x_y.a-b", config=c) == {"c_d": 1}
        assert config.get("x_y.e_f", config=c) == 2


@pytest.mark.parametrize("key", ["custom_key", "custom-key"])
def test_get_set_roundtrip(key):
    value = 123
    with config.set({key: value}):
        assert config.get("custom_key") == value
        assert config.get("custom-key") == value


def test_merge_None_to_dict():
    assert config.merge({"a": None, "c": 0}, {"a": {"b": 1}}) == {
        "a": {"b": 1},
        "c": 0,
    }


def test_deprecations():
    config.deprecations.deprecations['old_key'] = 'new.key'
    with pytest.warns(Warning) as info:
        with config.set(old_key=123):
            assert config.get("new.key") == 123

    assert "new.key" in str(info[0].message)


def test_core_file():
    """Test for default keys on our napari.yaml file."""
    # assert "temporary-directory" in napari.config.config


def test_sync(tmp_path):
    """Test that we can sync napari.config to a yaml file"""
    dest = tmp_path / 'dest.yaml'
    assert not os.path.isfile(dest)

    d = {}
    config.set({"abc.x": 123, 'b': 'hi'}, config=d)
    assert config.sync(d, destination=dest)
    assert os.path.isfile(dest)

    # make sure the yaml file matches the config
    with open(dest) as f:
        assert yaml.safe_load(f) == {'abc': {'x': 123}, 'b': 'hi'}

    # change the config and confirm it hasn't changed on disk
    config.set(b=10, config=d)
    with open(dest) as f:
        assert yaml.safe_load(f) == {'abc': {'x': 123}, 'b': 'hi'}

    # sync the config and make sure b has updated on disk
    assert config.sync(d, destination=dest)
    with open(dest) as f:
        assert yaml.safe_load(f) == {'abc': {'x': 123}, 'b': 10}

    # calling sync again should do nothing
    assert not config.sync(d, destination=dest)


def test_sync_with_serialization_errors(tmp_path, caplog):
    dest = tmp_path / 'dest.yaml'
    d = {}
    config.set(
        {'b': 10, 'i': Interpolation.NEAREST, 'strange': object()}, config=d,
    )
    config.sync(d, destination=dest)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "Error serializing object" in record.message
    assert record.levelname == "ERROR"

    with open(dest, 'r') as f:
        assert f.read() == "b: 10\ni: nearest\nstrange: <unserializeable>\n"


def test_sync_without_sync_status(tmp_path, caplog):
    dest = tmp_path / 'dest.yaml'
    d = {}
    config.set({"abc.x": 123, 'b': 10}, config=d)
    assert config.sync(d, destination=dest)
    assert os.path.isfile(dest)
    assert not config.sync(d, destination=dest)


def test_register_listener():
    """Test that we can register callbacks to listen to changes"""

    def callback(x):
        assert x == 5

    config.register_listener('some.key', callback)
    config.set({'some.key': 5})
    # make sure the callback is actually being called
    with pytest.raises(AssertionError):
        config.set({'some.key': 7})
