"""Tests locker functionality"""
import pytest

from napari.utils.lock import (
    AttributeNotFound,
    Lock,
    Locker,
    LockMode,
    ValueNotCompatibleWithLockMode,
)


def test_locker_initialization():
    """Tests locker initialization"""
    locker_dictionary = {"attribute1": Lock(), "attribute2": Lock()}
    locker = Locker(locker_dictionary)
    assert locker.has_attribute("attribute1")
    assert len(locker.list_locks()) == 2

    locker = Locker()
    assert not locker.has_attribute("attribute1")
    assert len(locker.list_locks()) == 0


def test_adding_locks():
    """Tests adding locks"""
    locker = Locker()
    locker.add_lock("attribute1")
    locker.add_lock("attribute2", [3, 4], False)
    assert locker.has_attribute("attribute1")
    assert len(locker.list_locks()) == 2


def test_locking():
    """Tests locking mechanism"""
    locker = Locker()
    locker.add_lock("attribute1")
    assert locker.is_locked("attribute1") == True
    assert locker.is_locked("attribute1", hard_lock=False) == True
    locker.unlock("attribute1")
    assert locker.is_locked("attribute1") == False
    assert locker.is_locked("attribute1", hard_lock=False) == False
    locker.lock("attribute1")
    assert locker.is_locked("attribute1") == True
    assert locker.is_locked("attribute1", hard_lock=False) == True

    assert locker.is_locked("attribute2") == False
    with pytest.raises(AttributeNotFound):
        locker.unlock("attribute2")

    with pytest.raises(AttributeNotFound):
        locker.lock("attribute2")

    locker.add_lock("attribute2", hard_lock=False)
    assert locker.is_locked("attribute2", hard_lock=True) == False
    assert locker.is_locked("attribute2", hard_lock=False) == True
    locker.unlock("attribute2")
    assert locker.is_locked("attribute2", hard_lock=True) == False
    assert locker.is_locked("attribute2", hard_lock=True) == False
    locker.lock("attribute2")
    assert locker.is_locked("attribute2", hard_lock=True) == False
    assert locker.is_locked("attribute2", hard_lock=False) == True


def test_in_list_lock_mode():
    """Testing in list lock mode"""
    locker = Locker()
    locker.add_lock(
        "attribute1",
        value=[1, 2, 3],
        hard_lock=True,
        value_lock_mode=LockMode.IN_LIST,
    )
    is_locked = locker.is_locked("attribute1", value=3)
    assert is_locked == False
    is_locked = locker.is_locked("attribute1", value=4)
    assert is_locked == True


def test_in_range_lock_mode():
    """Testing in range lock mode"""
    locker = Locker()
    locker.add_lock(
        "attribute1",
        value=[0, 10],
        hard_lock=True,
        value_lock_mode=LockMode.IN_RANGE,
    )
    assert locker.is_locked("attribute1", value=3) == False
    assert locker.is_locked("attribute1", value=11) == True


def test_inequality_lock_modes():
    """Testing in range lock mode"""
    locker = Locker()
    locker.add_lock(
        "attribute1",
        value=10,
        hard_lock=True,
        value_lock_mode=LockMode.SMALLER_THAN,
    )
    locker.add_lock(
        "attribute2",
        value=10,
        hard_lock=True,
        value_lock_mode=LockMode.LARGER_THAN,
    )
    assert locker.is_locked("attribute1", value=3) == False
    assert locker.is_locked("attribute1", value=11) == True
    assert locker.is_locked("attribute2", value=11) == False
    assert locker.is_locked("attribute2", value=3) == True


def test_value_mode_errors():
    """Testing in range lock mode"""
    locker = Locker()
    with pytest.raises(ValueNotCompatibleWithLockMode):
        locker.add_lock(
            "attribute1", value='10', value_lock_mode=LockMode.SMALLER_THAN
        )
    with pytest.raises(ValueNotCompatibleWithLockMode):
        locker.add_lock(
            "attribute1", value='10', value_lock_mode=LockMode.LARGER_THAN
        )
    with pytest.raises(ValueNotCompatibleWithLockMode):
        locker.add_lock(
            "attribute1", value='10', value_lock_mode=LockMode.IN_LIST
        )
    with pytest.raises(ValueNotCompatibleWithLockMode):
        locker.add_lock(
            "attribute1", value='10', value_lock_mode=LockMode.IN_RANGE
        )


def test_locks_dictionary_update():
    """Testing the update mechanism using dictionaries"""
    locker = Locker()
    locker_data = dict()
    locker_data['attribute1'] = {"value": 3, "locked": False}

    locker.update_locks(locker_data)
    assert locker.get_lock('attribute1').value == 3
    assert locker.get_lock('attribute1').locked == False

    locker_data['attribute1'] = {"value": 4}
    locker_data['attribute2'] = {"value": 6}
    locker.update_locks(locker_data)
    assert locker.get_lock('attribute1').value == 4
    assert locker.get_lock('attribute1').locked == False
    assert locker.get_lock('attribute2').value == 6
    assert locker.get_lock('attribute2').locked == True
