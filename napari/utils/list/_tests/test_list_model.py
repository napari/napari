from napari.utils.list import ListModel
from unittest.mock import Mock
import pytest


@pytest.fixture
def mocked_integer_list_model():
    """Create an integer list model with events mocked.

    Returns
    -------
    my_list : ListModel
        Integer list model.
    numbers : List
        Values of integers in list.
    mock_added : unittest.mock.Mock
        Mock connected to the added event.
    mock_removed : unittest.mock.Mock
        Mock connected to the removed event.
    mock_reordered : unittest.mock.Mock
        Mock connected to the reordered event.
    mock_changed : unittest.mock.Mock
        Mock connected to the changed event.
    """
    numbers = list(range(6))
    my_list = ListModel(int, numbers)

    mock_added = Mock()
    mock_removed = Mock()
    mock_reordered = Mock()
    mock_changed = Mock()
    my_list.events.added.connect(lambda e: mock_added.method())
    my_list.events.removed.connect(lambda e: mock_removed.method())
    my_list.events.reordered.connect(lambda e: mock_reordered.method())
    my_list.events.changed.connect(lambda e: mock_changed.method())

    mock = {
        'added': mock_added,
        'removed': mock_removed,
        'reordered': mock_reordered,
        'changed': mock_changed,
    }

    return my_list, numbers, mock


def test_create_integer_list_model():
    """Create a list model from list of integers."""
    numbers = list(range(6))
    my_list = ListModel(int, numbers)

    values = [i for i in my_list]
    assert values == numbers


def test_create_string_list_model_with_lookup():
    """Create a list model from list from list of strings."""
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    my_list = ListModel(str, letters, lookup={str: lambda q, e: q == e})

    values = [i for i in my_list]
    assert values == letters
    assert my_list['a'] == 'a'


def test_rearrange_string_list_model_with_lookup():
    """Create a list model from list from list of strings."""
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    my_list = ListModel(str, letters, lookup={str: lambda q, e: q == e})

    my_list['a', 'b'] = my_list['b', 'a']

    values = [i for i in my_list]
    assert values == ['b', 'a'] + letters[2:]


def test_append_integer_list(mocked_integer_list_model):
    """Append to a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list.append(7)
    values = [i for i in my_list]
    assert values == numbers + [7]
    mock['added'].method.assert_called_once()
    mock['removed'].method.assert_not_called()
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()


def test_insert_integer_list(mocked_integer_list_model):
    """Insert into a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list.insert(3, 7)
    values = [i for i in my_list]
    assert values == numbers[:3] + [7] + numbers[3:]
    mock['added'].method.assert_called_once()
    mock['removed'].method.assert_not_called()
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()
    mock['added'].reset_mock()
    mock['changed'].reset_mock()

    my_list.insert(0, 8)
    values = [i for i in my_list]
    assert values == [8] + numbers[:3] + [7] + numbers[3:]
    mock['added'].method.assert_called_once()
    mock['removed'].method.assert_not_called()
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()


def test_pop_integer_list(mocked_integer_list_model):
    """Pop from a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    value = my_list.pop(3)
    values = [i for i in my_list]
    assert values == numbers[:3] + numbers[4:]
    assert value == 3
    mock['added'].method.assert_not_called()
    mock['removed'].method.assert_called_once()
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()


def test_remove_integer_list(mocked_integer_list_model):
    """Remove from a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list.remove(3)
    values = [i for i in my_list]
    assert values == numbers[:3] + numbers[4:]
    mock['added'].method.assert_not_called()
    mock['removed'].method.assert_called_once()
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()


def test_clear_integer_list(mocked_integer_list_model):
    """Clear a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list.clear()
    values = [i for i in my_list]
    assert values == []
    mock['added'].method.assert_not_called()
    assert mock['removed'].method.call_count == len(numbers)
    mock['reordered'].method.assert_not_called()
    mock['changed'].method.assert_called_once()


def test_reorder_integer_list_with_swap(mocked_integer_list_model):
    """Reorder a list model with a swap."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list[0, 2] = my_list[2, 0]

    values = [i for i in my_list]
    assert values == [numbers[2], numbers[1], numbers[0]] + numbers[3:]
    mock['added'].method.assert_not_called()
    mock['added'].method.assert_not_called()
    mock['reordered'].method.assert_called_once()
    mock['changed'].method.assert_called_once()


def test_reorder_integer_list_with_tuple(mocked_integer_list_model):
    """Reorder a list model with a tuple."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list[(0, 1, 2)] = my_list[(1, 2, 0)]

    values = [i for i in my_list]
    assert values == [numbers[1], numbers[2], numbers[0]] + numbers[3:]
    mock['added'].method.assert_not_called()
    mock['added'].method.assert_not_called()
    mock['reordered'].method.assert_called_once()
    mock['changed'].method.assert_called_once()


def test_reverse_integer_list(mocked_integer_list_model):
    """Reverse a list model."""
    my_list, numbers, mock = mocked_integer_list_model

    my_list.reverse()
    values = [i for i in my_list[::-1]]
    assert values == numbers
    mock['added'].method.assert_not_called()
    mock['added'].method.assert_not_called()
    mock['reordered'].method.assert_called_once()
    mock['changed'].method.assert_called_once()
