import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.text_manager import TextManager


def test_empty_text_manager_property():
    """Test creating an empty text manager in property mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence', n_text=0, properties=properties
    )
    assert text_manager.text.array.size == 0

    # add a text element
    properties['confidence'] = np.array([0.5])
    text_manager.add(1)
    np.testing.assert_equal(text_manager.text.array, ['0.5'])


def test_add_many_text_property():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence',
        n_text=0,
        properties=properties,
    )

    properties['confidence'] = np.array([0.5, 0.5])
    text_manager.add(2)

    np.testing.assert_equal(text_manager.text.array, ['0.5'] * 2)


def test_empty_text_manager_format():
    """Test creating an empty text manager in formatted mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text = 'confidence: {confidence:.2f}'
    text_manager = TextManager(text=text, n_text=0, properties=properties)
    assert text_manager.text.array.size == 0

    # add a text element
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    np.testing.assert_equal(text_manager.text.array, ['confidence: 0.50'])


def test_add_many_text_formatted():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence: {confidence:.2f}',
        n_text=0,
        properties=properties,
    )

    properties['confidence'] = np.append(properties['confidence'], [0.5] * 2)
    text_manager.add(2)

    np.testing.assert_equal(text_manager.text.array, ['confidence: 0.50'] * 2)


def test_text_manager_property():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.text.array, classes)

    # add new text with properties
    properties['class'] = np.concatenate((classes, ['A']))
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    np.testing.assert_equal(text_manager.text.array, properties['class'])

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.text.array, properties['class'][1::])


def test_text_manager_format():
    n_text = 3
    text = 'confidence: {confidence:.2f}'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    expected_text = np.array(
        ['confidence: 0.50', 'confidence: 0.30', 'confidence: 1.00']
    )
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.text.array, expected_text)

    # add new text with properties
    properties['class'] = np.concatenate((classes, ['A']))
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    expected_text_2 = np.concatenate([expected_text, ['confidence: 0.50']])
    np.testing.assert_equal(text_manager.text.array, expected_text_2)

    # test getting the text elements when there are none in view
    text_view = text_manager.view_text([])
    np.testing.assert_equal(text_view, [''])

    # test getting the text elements when the first two elements are in view
    text_view = text_manager.view_text([0, 1])
    np.testing.assert_equal(text_view, expected_text_2[0:2])

    text_manager.anchor = 'center'
    coords = np.array([[0, 0], [10, 10], [20, 20]])
    text_coords = text_manager.compute_text_coords(coords, ndisplay=3)
    np.testing.assert_equal(text_coords, (coords, 'center', 'center'))

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.text.array, expected_text_2[1::])


def test_text_manager_invalid_format_string():
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}
    with pytest.raises(ValidationError):
        TextManager(text=text, n_text=3, properties=properties)


def test_text_manager_format_string_contains_non_property():
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}
    with pytest.raises(ValidationError):
        TextManager(text=text, n_text=3, properties=properties)


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
    text_manager.refresh_text(new_properties, n_text)
    np.testing.assert_equal(new_classes, text_manager.text.array)


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


def test_text_with_invalid_format_string_then_raises_on_validation():
    n_text = 3
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.raises(ValidationError):
        TextManager(text=text, n_text=n_text, properties=properties)


def test_text_with_format_string_missing_property_then_raises_on_validation():
    n_text = 3
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.raises(ValidationError):
        TextManager(text=text, n_text=n_text, properties=properties)


def test_text_constant():
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}

    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )

    np.testing.assert_array_equal(text_manager.text.array, ['point'] * n_text)


def test_text_constant_with_empty_properties():
    n_text = 3
    text_manager = TextManager(text='point', n_text=n_text, properties={})
    np.testing.assert_array_equal(text_manager.text.array, ['point'] * n_text)


def test_add_with_text_constant():
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )
    assert len(text_manager.text.array) == n_text

    properties['class'] = np.append(properties['class'], ['C', 'C'])
    text_manager.add(2)

    np.testing.assert_array_equal(text_manager.text.array, ['point'] * 5)


def test_add_with_text_constant_init_empty():
    properties = {'class': np.empty((0,), dtype=str)}
    text_manager = TextManager(text='point', n_text=0, properties=properties)

    properties['class'] = np.append(properties['class'], ['C', 'C'])
    text_manager.add(2)

    np.testing.assert_array_equal(text_manager.text.array, ['point'] * 2)


def test_remove_with_text_constant():
    n_text = 5
    properties = {'class': np.array(['A', 'B', 'C', 'D', 'E'])}
    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )

    text_manager.remove([1, 3])

    np.testing.assert_equal(text_manager.text.array, ['point'] * 3)


def test_text_direct():
    values = ['one', 'two', 'three']
    text_manager = TextManager(text=values, n_text=3, properties={})
    np.testing.assert_array_equal(text_manager.text.array, values)


def test_text_direct_add():
    values = ['one', 'two', 'three']
    text_manager = TextManager(text=values, n_text=3, properties={})

    text_manager.add(2)

    np.testing.assert_array_equal(
        text_manager.text.array, ['one', 'two', 'three', '', '']
    )


def test_text_direct_remove():
    values = ['one', 'two', 'three', 'four']
    text_manager = TextManager(text=values, n_text=4, properties={})

    text_manager.remove([1, 3])

    np.testing.assert_array_equal(text_manager.text.array, ['one', 'three'])


def test_text_direct_deprecated_values_field():
    values = ['one', 'two', 'three']
    with pytest.warns(DeprecationWarning):
        text_manager = TextManager(values=values, n_text=3, properties={})
        np.testing.assert_array_equal(text_manager.values, values)


def test_text_direct_set_deprecated_values_field():
    values = ['one', 'two', 'three']
    text_manager = TextManager(text=values, n_text=3, properties={})

    new_values = ['four', 'five', 'six']
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(text_manager.values, values)
        text_manager.values = new_values
        np.testing.assert_array_equal(text_manager.values, new_values)
