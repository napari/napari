from napari.utils.context import ContextKeyService


def test_root_service():
    root = ContextKeyService()
    assert root.context_id == 0

    assert root.dict() == dict(root) == {}
    _id = id(root._my_context)

    # set
    root['key'] = 1

    # get
    assert root['key'] == 1
    assert dict(root) == {'key': 1}
    assert id(root._my_context) == _id
    assert len(root) == 1
    assert 'key' in root

    # update
    root['key'] = 2
    assert dict(root) == {'key': 2}
    assert id(root._my_context) == _id

    # delete
    del root['key']
    assert dict(root) == {}
    assert id(root._my_context) == _id

    D = {"a": 1, "b": 2}
    root.update(D)
    assert dict(root) == D
    assert repr(root)

    root.clear()
    assert not root


def test_scoped_service_inherits():
    root = ContextKeyService()
    scoped = root.create_scoped(None)

    # add to the scoped dict ... the root is unaffected
    scoped['k'] = 0
    assert dict(root) == {}
    assert dict(scoped) == {'k': 0}

    assert repr(scoped)

    # add to the root dict ... the scoped dict is unaffected
    root['k'] = 1
    assert dict(root) == {'k': 1}
    assert dict(scoped) == {'k': 0}
    assert scoped['k'] == 0

    # delete the scoped key, now it inherits from the parent
    del scoped['k']
    assert dict(root) == {'k': 1}
    assert dict(scoped) == {'k': 1}
    assert scoped['k'] == 1

    # deleting again has no consequence
    del scoped['k']
    assert dict(scoped) == {'k': 1}

    # deleting from root affects them
    scoped['k2'] = 10
    root.clear()
    assert dict(root) == {}
    assert dict(scoped) == {'k2': 10}
