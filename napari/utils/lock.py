"""
A generic lock class to be used as a locking mechanism.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator

from napari.utils.translations import trans


class LockMode(Enum):
    """Lock modes"""

    EXACT = 0
    IN_LIST = 1
    IN_RANGE = 2
    LARGER_THAN = 3
    SMALLER_THAN = 4


class ValueNotCompatibleWithLockMode(Exception):
    """Custom error that is raised when the value is not compatible with lock mode"""

    def __init__(self, value, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)


class AttributeNotFound(Exception):
    """Custom error that is raised when the attribute is not found in the locker"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AttributeAlreadyAdded(Exception):
    """Custom error that is raised when the atribute is already found in the locker"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class Lock(BaseModel):
    """
    A lock consists of components:
    1. A boolean to house the status of the lock (open, locked).
    2. A value of the lock, for example, an attribute maximum can be locked to certain number.
    3. A lock mode to desginate the type of lock (soft (user), or hard (code and user))
    4. comments
    """

    value_lock_mode: LockMode = LockMode.EXACT
    value: Any = None
    locked: bool = True
    hard_lock: bool = True
    comments: Optional[str] = None

    @validator("value_lock_mode")
    def value_lock_mode_lock_mode_valid(cls, value):
        """Validate value lock mode"""
        return value

    @validator("value")
    def value_lock_mode_valid(cls, value, values, **kwargs):
        """Validate value"""
        value_lock_mode = values["value_lock_mode"]

        if (
            value_lock_mode in (LockMode.LARGER_THAN, LockMode.SMALLER_THAN)
        ) and not type(value) in (int, float):
            raise ValueNotCompatibleWithLockMode(
                value=value,
                message=trans._(
                    "Value should be numeric for lock modes: LARGER_THAN and SMALLER_THAN",
                    deferred=True,
                ),
            )

        if value_lock_mode == LockMode.IN_LIST and type(value) not in (
            list,
            tuple,
        ):
            raise ValueNotCompatibleWithLockMode(
                value=value,
                message=trans._(
                    "Value should be list or tuple for lock mode: IN_LIST",
                    deferred=True,
                ),
            )

        if value_lock_mode == LockMode.IN_RANGE and (
            type(value) not in (list, tuple) or len(value) != 2
        ):
            raise ValueNotCompatibleWithLockMode(
                value=value,
                message=trans._(
                    "Value should be list or tuple of 2 elements for lock mode: IN_RANGE",
                    deferred=True,
                ),
            )

        return value

    def lock(self) -> None:
        """locks the lock"""
        self.locked = True

    def unlock(self) -> None:
        """locks the lock"""
        self.locked = False

    def is_valid_value(self, value: Any) -> bool:
        """Checks if value is permitted in the lock"""
        if self.value_lock_mode == LockMode.EXACT:
            return False
        elif self.value_lock_mode == LockMode.IN_LIST:
            return value in self.value
        elif self.value_lock_mode == LockMode.IN_RANGE:
            return value >= self.value[0] and value <= self.value[1]
        elif self.value_lock_mode == LockMode.SMALLER_THAN:
            return value <= self.value
        # elif self.value_lock_mode == LockMode.LARGER_THAN:
        return value >= self.value

    def is_locked(self, hard_lock: bool = True, value: Any = None):
        """Check if there is a lock on a specific attribute"""
        if not self.locked:
            return False

        if self.hard_lock is True or hard_lock is False:
            if self.is_valid_value(value):
                return False
            else:
                return True

        return False


class Locker:
    """
    A generic class to be used as a locking mechanism. It contains a collection of locks
    for different attributes.
    """

    _lock_dictionary: Dict[str, Lock]

    def __init__(self, lock_dictionary: Optional[Dict[str, Lock]] = None):
        """Initializes the locker"""
        if lock_dictionary is not None:
            self._lock_dictionary = lock_dictionary
        else:
            self._lock_dictionary = {}

    def add_lock(
        self,
        attribute: str,
        value: Any = None,
        hard_lock: bool = True,
        locked: bool = True,
        value_lock_mode: LockMode = LockMode.EXACT,
        comments: Optional[str] = None,
    ) -> None:
        """Adds a lock to the locker"""
        if self.has_attribute(attribute):
            raise AttributeAlreadyAdded(
                trans._(
                    "Attribute is already added to the locker", deferred=True
                )
            )

        self._lock_dictionary[attribute] = Lock(
            value=value,
            hard_lock=hard_lock,
            value_lock_mode=value_lock_mode,
            comments=comments,
            locked=locked,
        )

    def is_locked(
        self, attribute: str, hard_lock: bool = True, value: Any = None
    ) -> bool:
        """Check if there is a lock on a specific attribute"""
        if not self.has_attribute(attribute):
            return False

        lock = self._lock_dictionary[attribute]
        return lock.is_locked(hard_lock=hard_lock, value=value)

    def has_attribute(self, attribute: str) -> bool:
        """Checks if the lock has a specific attribute"""
        return attribute in self._lock_dictionary

    def lock(self, attribute: str) -> None:
        """Locks a specifc attribute"""
        if not self.has_attribute(attribute):
            raise AttributeNotFound(
                trans._("Attribute not found in the locker", deferred=True)
            )
        self.get_lock(attribute).lock()

    def unlock(self, attribute: str) -> None:
        """Unlocks a specific attribute"""
        if not self.has_attribute(attribute):
            raise AttributeNotFound(
                trans._("Attribute not found in the locker", deferred=True)
            )
        self.get_lock(attribute).unlock()

    def list_locks(self) -> List[str]:
        """Returns the list of locks available"""
        return list(self._lock_dictionary.keys())

    def get_lock(self, attribute: str) -> Lock:
        """Returns the lock value"""
        return self._lock_dictionary[attribute]

    def update_locks(self, locker_dictionary_update: Dict):
        """Update the locks using a dictionary"""
        for attribute, lock_data in locker_dictionary_update.items():
            if attribute in self._lock_dictionary.keys():
                current_lock = self.get_lock(attribute)
                updated_lock = current_lock.copy(update=lock_data)
            else:
                updated_lock = Lock(**lock_data)

            self._lock_dictionary[attribute] = updated_lock
