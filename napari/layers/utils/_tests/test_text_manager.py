import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    FormatStringEncoding,
    ManualStringEncoding,
)
from napari.layers.utils.text_manager import TextManager


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_empty_text_manager_property():
    """Test creating an empty text manager in property mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence', n_text=0, properties=properties
    )
    assert text_manager.values.size == 0

    # add a text element
    new_properties = {'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    np.testing.assert_equal(text_manager.values, ['0.5'])


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_add_many_text_property():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence',
        n_text=0,
        properties=properties,
    )

    text_manager.add({'confidence': np.array([0.5])}, 2)

    np.testing.assert_equal(text_manager.values, ['0.5'] * 2)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_empty_text_manager_format():
    """Test creating an empty text manager in formatted mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text = 'confidence: {confidence:.2f}'
    text_manager = TextManager(text=text, n_text=0, properties=properties)
    assert text_manager.values.size == 0

    # add a text element
    new_properties = {'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    np.testing.assert_equal(text_manager.values, ['confidence: 0.50'])


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_add_many_text_formatted():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence: {confidence:.2f}',
        n_text=0,
        properties=properties,
    )

    text_manager.add({'confidence': np.array([0.5])}, 2)

    np.testing.assert_equal(text_manager.values, ['confidence: 0.50'] * 2)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_manager_property():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.values, classes)

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    expected_text_2 = np.concatenate([classes, ['A']])
    np.testing.assert_equal(text_manager.values, expected_text_2)

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.values, expected_text_2[1::])


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_manager_format():
    n_text = 3
    text = 'confidence: {confidence:.2f}'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    expected_text = np.array(
        ['confidence: 0.50', 'confidence: 0.30', 'confidence: 1.00']
    )
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.values, expected_text)

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    expected_text_2 = np.concatenate([expected_text, ['confidence: 0.50']])
    np.testing.assert_equal(text_manager.values, expected_text_2)

    # test getting the text elements when there are none in view
    text_view = text_manager.view_text([])
    np.testing.assert_equal(text_view, np.empty((0,), dtype=str))

    # test getting the text elements when the first two elements are in view
    text_view = text_manager.view_text([0, 1])
    np.testing.assert_equal(text_view, expected_text_2[0:2])

    text_manager.anchor = 'center'
    coords = np.array([[0, 0], [10, 10], [20, 20]])
    text_coords = text_manager.compute_text_coords(coords, ndisplay=3)
    np.testing.assert_equal(text_coords, (coords, 'center', 'center'))

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.values, expected_text_2[1::])


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_refresh_text():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)

    new_classes = np.array(['D', 'E', 'F'])
    new_properties = {
        'class': new_classes,
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager.refresh_text(new_properties)
    np.testing.assert_equal(new_classes, text_manager.values)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_equality():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager_1 = TextManager(
        text=text, n_text=n_text, properties=properties, color='red'
    )
    text_manager_2 = TextManager(
        text=text, n_text=n_text, properties=properties, color='red'
    )

    assert text_manager_1 == text_manager_2
    assert not (text_manager_1 != text_manager_2)

    text_manager_2.color = 'blue'
    assert text_manager_1 != text_manager_2
    assert not (text_manager_1 == text_manager_2)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_blending_modes():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(
        text=text,
        n_text=n_text,
        properties=properties,
        color='red',
        blending='translucent',
    )
    assert text_manager.blending == 'translucent'

    # set to another valid blending mode
    text_manager.blending = 'additive'
    assert text_manager.blending == 'additive'

    # set to opaque, which is not allowed
    with pytest.warns(RuntimeWarning):
        text_manager.blending = 'opaque'
        assert text_manager.blending == 'translucent'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_with_invalid_format_string_then_fallback_with_warning():
    n_text = 3
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.warns(RuntimeWarning):
        text_manager = TextManager(
            text=text, n_text=n_text, properties=properties
        )

    np.testing.assert_array_equal(text_manager.values, [''] * n_text)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_with_format_string_missing_property_then_fallback_with_warning():
    n_text = 3
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.warns(RuntimeWarning):
        text_manager = TextManager(
            text=text, n_text=n_text, properties=properties
        )

    np.testing.assert_array_equal(text_manager.values, [''] * n_text)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_constant_then_repeat_values():
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}

    text_manager = TextManager(
        text={'constant': 'point'}, n_text=n_text, properties=properties
    )

    np.testing.assert_array_equal(text_manager.values, ['point'] * n_text)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_text_constant_with_no_properties():
    text_manager = TextManager(text={'constant': 'point'}, n_text=3)
    np.testing.assert_array_equal(text_manager.values, ['point'] * 3)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_add_with_text_constant():
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(
        text={'constant': 'point'}, n_text=n_text, properties=properties
    )
    np.testing.assert_array_equal(text_manager.values, ['point'] * 3)

    text_manager.add({'class': np.array(['C'])}, 2)

    np.testing.assert_array_equal(text_manager.values, ['point'] * 5)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_add_with_text_constant_init_empty():
    properties = {}
    text_manager = TextManager(
        text={'constant': 'point'}, n_text=0, properties=properties
    )

    text_manager.add({'class': np.array(['C'])}, 2)

    np.testing.assert_array_equal(text_manager.values, ['point'] * 2)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_remove_with_text_constant_then_ignored():
    n_text = 5
    properties = {'class': np.array(['A', 'B', 'C', 'D', 'E'])}
    text_manager = TextManager(
        text={'constant': 'point'}, n_text=n_text, properties=properties
    )

    text_manager.remove([1, 3])

    np.testing.assert_array_equal(text_manager.values, ['point'] * n_text)


def test_from_layer():
    text = {
        'string': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    features = pd.DataFrame(
        {
            'class': np.array(['A', 'B', 'C']),
            'confidence': np.array([1, 0.5, 0]),
        }
    )
    text_manager = TextManager._from_layer(
        text=text,
        features=features,
    )

    np.testing.assert_array_equal(text_manager.values, ['A', 'B', 'C'])
    np.testing.assert_array_equal(text_manager.translation, [-0.5, 1])
    assert not text_manager.visible


def test_update_from_layer():
    text = {
        'string': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    features = pd.DataFrame(
        {
            'class': ['A', 'B', 'C'],
            'confidence': [1, 0.5, 0],
        }
    )
    text_manager = TextManager._from_layer(
        text=text,
        features=features,
    )

    text = {
        'string': 'Conf: {confidence:.2f}',
        'translation': [1.5, -2],
        'size': 9000,
    }
    text_manager._update_from_layer(text=text, features=features)

    np.testing.assert_array_equal(
        text_manager.values, ['Conf: 1.00', 'Conf: 0.50', 'Conf: 0.00']
    )
    np.testing.assert_array_equal(text_manager.translation, [1.5, -2])
    assert text_manager.visible
    assert text_manager.size == 9000


def test_update_from_layer_with_invalid_value_fails_safely():
    features = pd.DataFrame(
        {
            'class': ['A', 'B', 'C'],
            'confidence': [1, 0.5, 0],
        }
    )
    text_manager = TextManager._from_layer(
        text='class',
        features=features,
    )
    before = text_manager.copy(deep=True)

    text = {
        'string': 'confidence',
        'size': -3,
    }

    with pytest.raises(ValidationError):
        text_manager._update_from_layer(text=text, features=features)

    assert text_manager == before


def test_update_from_layer_with_warning_only_one_emitted():
    features = pd.DataFrame({'class': ['A', 'B', 'C']})
    text_manager = TextManager._from_layer(
        text='class',
        features=features,
    )

    text = {
        'string': 'class',
        'blending': 'opaque',
    }

    with pytest.warns(RuntimeWarning) as record:
        text_manager._update_from_layer(
            text=text,
            features=features,
        )

    assert len(record) == 1


def test_init_with_no_string():
    text_manager = TextManager(string=None)

    assert text_manager.string == ConstantStringEncoding(constant='')
    np.testing.assert_array_equal(text_manager.values, '')


def test_init_with_constant_string():
    text_manager = TextManager(string={'constant': 'A'})

    assert text_manager.string == ConstantStringEncoding(constant='A')
    np.testing.assert_array_equal(text_manager.values, 'A')


def test_init_with_manual_string():
    features = pd.DataFrame(index=range(3))

    text_manager = TextManager(string=['A', 'B', 'C'], features=features)

    assert text_manager.string == ManualStringEncoding(array=['A', 'B', 'C'])
    np.testing.assert_array_equal(text_manager.values, ['A', 'B', 'C'])


def test_init_with_format_string():
    features = pd.DataFrame({'class': ['A', 'B', 'C']})

    text_manager = TextManager(string='class: {class}', features=features)

    assert text_manager.string == FormatStringEncoding(format='class: {class}')
    np.testing.assert_array_equal(
        text_manager.values, ['class: A', 'class: B', 'class: C']
    )


def test_apply_with_constant_string():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string={'constant': 'A'})

    features = pd.DataFrame(index=range(5))
    text_manager.apply(features)

    np.testing.assert_array_equal(text_manager.values, 'A')


def test_apply_with_manual_string():
    string = {
        'array': ['A', 'B', 'C'],
        'default': 'D',
    }
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string=string, features=features)

    features = pd.DataFrame(index=range(5))
    text_manager.apply(features)

    np.testing.assert_array_equal(
        text_manager.values, ['A', 'B', 'C', 'D', 'D']
    )


def test_apply_with_derived_string():
    features = pd.DataFrame({'class': ['A', 'B', 'C']})
    text_manager = TextManager(string='class: {class}', features=features)

    features = pd.DataFrame({'class': ['A', 'B', 'C', 'D', 'E']})
    text_manager.apply(features)

    np.testing.assert_array_equal(
        text_manager.values,
        ['class: A', 'class: B', 'class: C', 'class: D', 'class: E'],
    )


def test_refresh_with_constant_string():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string={'constant': 'A'})

    text_manager.string = {'constant': 'B'}
    text_manager.refresh(features)

    np.testing.assert_array_equal(text_manager.values, 'B')


def test_refresh_with_manual_string():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string=['A', 'B', 'C'], features=features)

    text_manager.string = ['C', 'B', 'A']
    text_manager.refresh(features)

    np.testing.assert_array_equal(text_manager.values, ['C', 'B', 'A'])


def test_refresh_with_derived_string():
    features = pd.DataFrame({'class': ['A', 'B', 'C']})
    text_manager = TextManager(string='class: {class}', features=features)

    features = pd.DataFrame({'class': ['E', 'D', 'C', 'B', 'A']})
    text_manager.refresh(features)

    np.testing.assert_array_equal(
        text_manager.values,
        ['class: E', 'class: D', 'class: C', 'class: B', 'class: A'],
    )


def test_copy_paste_with_constant_string():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string={'constant': 'A'}, features=features)

    copied = text_manager._copy([0, 2])
    text_manager._paste(**copied)

    np.testing.assert_array_equal(text_manager.values, 'A')


def test_copy_paste_with_manual_string():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string=['A', 'B', 'C'], features=features)

    copied = text_manager._copy([0, 2])
    text_manager._paste(**copied)

    np.testing.assert_array_equal(
        text_manager.values, ['A', 'B', 'C', 'A', 'C']
    )


def test_copy_paste_with_derived_string():
    features = pd.DataFrame({'class': ['A', 'B', 'C']})
    text_manager = TextManager(string='class: {class}', features=features)

    copied = text_manager._copy([0, 2])
    text_manager._paste(**copied)

    np.testing.assert_array_equal(
        text_manager.values,
        ['class: A', 'class: B', 'class: C', 'class: A', 'class: C'],
    )


def test_serialization():
    features = pd.DataFrame(
        {'class': ['A', 'B', 'C'], 'confidence': [0.5, 0.3, 1]}
    )
    original = TextManager(features=features, string='class', color='red')

    serialized = original.dict()
    deserialized = TextManager(**serialized)

    assert original == deserialized


def test_view_text_with_constant_text():
    features = pd.DataFrame(index=range(3))
    text_manager = TextManager(string={'constant': 'A'}, features=features)

    copied = text_manager._copy([0, 2])
    text_manager._paste(**copied)

    actual = text_manager.view_text([0, 1])

    # view_text promises to return an Nx1 array, not just something
    # broadcastable to an Nx1, so explicitly check the length
    # because assert_array_equal broadcasts scalars automatically
    assert len(actual) == 2
    np.testing.assert_array_equal(actual, ['A', 'A'])
