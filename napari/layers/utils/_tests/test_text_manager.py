import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.text_manager import TextManager
from napari.utils.colormaps.standardize_color import transform_color


def test_empty_text_manager_property():
    """Test creating an empty text manager in property mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(string='confidence', properties=properties)
    assert len(text_manager.string._get_array(properties, 0)) == 0

    properties['confidence'] = np.array([0.5])
    string_array = text_manager.string._get_array(properties, 1)

    np.testing.assert_equal(string_array, ['0.5'])


def test_add_many_text_property():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(string='confidence', properties=properties)

    properties['confidence'] = np.array([0.5, 0.5])
    string_array = text_manager.string._get_array(properties, 2)

    np.testing.assert_equal(string_array, ['0.5'] * 2)


def test_empty_text_manager_format():
    """Test creating an empty text manager in formatted mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text = 'confidence: {confidence:.2f}'
    text_manager = TextManager(string=text, properties=properties)
    assert len(text_manager.string._get_array(properties, 0)) == 0

    properties['confidence'] = np.append(properties['confidence'], 0.5)
    string_array = text_manager.string._get_array(properties, 1)

    np.testing.assert_equal(string_array, ['confidence: 0.50'])


def test_add_many_text_formatted():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        string='confidence: {confidence:.2f}', properties=properties
    )

    properties['confidence'] = np.append(properties['confidence'], [0.5] * 2)
    string_array = text_manager.string._get_array(properties, 2)

    np.testing.assert_equal(string_array, ['confidence: 0.50'] * 2)


def test_text_manager_property():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager = TextManager(string='class', properties=properties)

    # add new text with properties
    properties['class'] = np.append(properties['class'], 'A')
    properties['confidence'] = np.append(properties['confidence'], 0.5)
    string_array = text_manager.string._get_array(properties, 4)
    np.testing.assert_equal(string_array, properties['class'])

    # remove the first text element
    properties['class'] = np.delete(properties['class'], 0)
    properties['confidence'] = np.delete(properties['confidence'], 0)
    text_manager.remove({0})
    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_equal(string_array, properties['class'])


def test_text_manager_format():
    text = 'confidence: {confidence:.2f}'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    expected_text = np.array(
        ['confidence: 0.50', 'confidence: 0.30', 'confidence: 1.00']
    )
    text_manager = TextManager(string=text, properties=properties)
    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_equal(string_array, expected_text)

    # add new text with properties
    properties['class'] = np.append(properties['class'], 'A')
    properties['confidence'] = np.append(properties['confidence'], 0.5)

    string_array = text_manager.string._get_array(properties, 4)
    expected_text_2 = np.append(expected_text, 'confidence: 0.50')
    np.testing.assert_equal(string_array, expected_text_2)

    # test getting the text elements when there are none in view
    string_array = text_manager.string._get_array(properties, 4, [])
    np.testing.assert_equal(string_array, np.empty((0,), dtype=str))

    # test getting the text elements when the first two elements are in view
    string_array = text_manager.string._get_array(properties, 4, [0, 1])
    np.testing.assert_equal(string_array, expected_text_2[0:2])

    text_manager.anchor = 'center'
    coords = np.array([[0, 0], [10, 10], [20, 20]])
    text_coords = text_manager.compute_text_coords(coords, ndisplay=3)
    np.testing.assert_equal(text_coords, (coords, 'center', 'center'))

    # remove the first text element
    properties['class'] = np.delete(properties['class'], 0)
    properties['confidence'] = np.delete(properties['confidence'], 0)
    text_manager.remove({0})
    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_equal(string_array, expected_text_2[1::])


def test_text_manager_invalid_format_string():
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}
    with pytest.raises(ValidationError):
        TextManager(string=text, properties=properties)


def test_text_manager_format_string_contains_non_property():
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}
    with pytest.raises(ValidationError):
        TextManager(string=text, properties=properties)


def test_refresh_text():
    text = 'class'
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager = TextManager(string=text, properties=properties)

    new_properties = {
        'class': np.array(['D', 'E', 'F']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager.refresh_text(new_properties)
    string_array = text_manager.string._get_array(new_properties, 3)
    np.testing.assert_equal(string_array, new_properties['class'])


def test_equality():
    text = 'class'
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager_1 = TextManager(
        string=text, properties=properties, color='red'
    )
    text_manager_2 = TextManager(
        string=text, properties=properties, color='red'
    )

    assert text_manager_1 == text_manager_2
    assert not (text_manager_1 != text_manager_2)

    text_manager_2.color = 'blue'
    assert text_manager_1 != text_manager_2
    assert not (text_manager_1 == text_manager_2)


def test_blending_modes():
    text = 'class'
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager = TextManager(
        string=text,
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
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.raises(ValidationError):
        TextManager(string=text, properties=properties)


def test_text_with_format_string_missing_property_then_raises_on_validation():
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    with pytest.raises(ValidationError):
        TextManager(string=text, properties=properties)


def test_text_constant():
    properties = {'class': np.array(['A', 'B', 'C'])}

    text_manager = TextManager(string='point', properties=properties)

    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_array_equal(string_array, ['point'] * 3)


def test_text_constant_with_empty_properties():
    properties = {}
    text_manager = TextManager(string='point', properties=properties)
    np.testing.assert_equal(
        text_manager.string._get_array(properties, 3), 'point'
    )


def test_add_with_text_constant():
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(string='point', properties=properties)
    np.testing.assert_equal(
        text_manager.string._get_array(properties, 3), 'point'
    )

    properties['class'] = np.append(properties['class'], ['C', 'C'])
    string_array = text_manager.string._get_array(properties, 5)

    np.testing.assert_equal(string_array, 'point')


def test_add_with_text_constant_init_empty():
    properties = {'class': np.empty((0,), dtype=str)}
    text_manager = TextManager(string='point', properties=properties)

    properties['class'] = np.append(properties['class'], ['C', 'C'])
    string_array = text_manager.string._get_array(properties, 2)

    np.testing.assert_array_equal(string_array, 'point')


def test_remove_with_text_constant():
    n_text = 5
    properties = {'class': np.array(['A', 'B', 'C', 'D', 'E'])}
    text_manager = TextManager(
        string='point', n_string=n_text, properties=properties
    )

    properties['class'] = np.delete(properties['class'], [1, 3])
    text_manager.remove([1, 3])
    string_array = text_manager.string._get_array(properties, 2)

    np.testing.assert_equal(string_array, 'point')


def test_text_direct():
    values = ['one', 'two', 'three']
    properties = {}
    text_manager = TextManager(string=values, properties=properties)
    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_array_equal(string_array, values)


def test_text_direct_remove():
    values = ['one', 'two', 'three', 'four']
    properties = {}
    text_manager = TextManager(string=values, properties=properties)

    text_manager.remove([1, 3])
    string_array = text_manager.string._get_array(properties, 2)

    np.testing.assert_array_equal(string_array, ['one', 'three'])


def test_text_direct_add_deprecated():
    values = ['one', 'two', 'three']
    text_manager = TextManager(string=values, properties={})

    with pytest.warns(DeprecationWarning):
        text_manager.add({}, num_to_add=2)
        string_array = text_manager.values

    np.testing.assert_array_equal(string_array, ['one', 'two', 'three'])


def test_text_format_deprecated_text_parameter():
    properties = {'class': np.array(['A', 'B', 'C'])}
    with pytest.warns(DeprecationWarning):
        text_manager = TextManager(
            text='class', n_text=3, properties=properties
        )
    string_array = text_manager.string._get_array(properties, 3)

    np.testing.assert_array_equal(string_array, properties['class'])


def test_text_direct_deprecated_values_field():
    values = ['one', 'two', 'three']
    with pytest.warns(DeprecationWarning):
        text_manager = TextManager(values=values, n_text=3, properties={})
        np.testing.assert_array_equal(text_manager.values, values)


def test_text_direct_set_deprecated_values_field():
    values = ['one', 'two', 'three']
    text_manager = TextManager(string=values, properties={})

    new_values = ['four', 'five', 'six']
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(text_manager.values, values)
        text_manager.values = new_values
        np.testing.assert_array_equal(text_manager.values, new_values)


def test_text_add_with_deprecated_properties_succeeds():
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(string='class', properties=properties)

    properties['class'] = np.append(properties['class'], ['C', 'C'])
    with pytest.warns(DeprecationWarning):
        text_manager.add({'class': np.array(['C'])}, 2)
        string_array = text_manager.values

    np.testing.assert_array_equal(string_array, properties['class'])


def test_multi_color_direct():
    classes = np.array(['A', 'B', 'C'])
    colors = np.array(['red', 'green', 'blue'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(
        string='class', properties=properties, color=colors
    )

    color_array = text_manager.color._get_array(properties, 3)
    _assert_colors_equal(color_array, colors)


def test_multi_color_property():
    colors = np.array(['red', 'green', 'blue'])
    properties = {'colors': colors, 'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(
        string='colors', properties=properties, color='colors'
    )

    color_array = text_manager.color._get_array(properties, 3)
    _assert_colors_equal(color_array, colors)


def test_multi_color_non_property():
    properties = {
        'class': np.array(['A', 'B', 'C']),
    }
    with pytest.raises(ValidationError):
        TextManager(string='class', properties=properties, color='class_color')


def test_multi_color_property_discrete_map():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0.3, 1]),
    }
    color = {
        'property': 'class',
        'categorical_colormap': {'A': 'red', 'B': 'green', 'C': 'blue'},
    }

    text_manager = TextManager(
        string='class', properties=properties, color=color
    )

    color_array = text_manager.color._get_array(properties, 3)
    _assert_colors_equal(color_array, ['red', 'green', 'blue'])


def test_multi_color_property_continuous_map():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0, 1]),
    }
    color = {
        'property': 'confidence',
        'colormap': 'gray',
    }

    text_manager = TextManager(
        string='class', properties=properties, color=color
    )

    color_array = text_manager.color._get_array(properties, 3)
    expected_color_array = transform_color([[0.5] * 3, [0] * 3, [1] * 3])
    _assert_colors_equal(color_array, expected_color_array)


def test_multi_color_property_continuous_map_with_contrast_limits():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0, -1.5, 2]),
    }
    color = {
        'property': 'confidence',
        'colormap': 'gray',
        'contrast_limits': [-1, 1],
    }

    text_manager = TextManager(
        string='class', properties=properties, color=color
    )

    color_array = text_manager.color._get_array(properties, 3)
    expected_color_array = transform_color([[0.5] * 3, [0] * 3, [1] * 3])
    _assert_colors_equal(color_array, expected_color_array)


def test_color_missing_field_errors():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0, 1]),
    }
    color = {
        'categorical_colormap': {'A': 'red', 'B': 'green', 'C': 'blue'},
    }

    with pytest.raises(ValidationError):
        TextManager(string='class', properties=properties, color=color)


def test_color_too_many_fields_use_first_matching():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([0.5, 0, 1]),
    }
    color = {
        'property': 'confidence',
        'categorical_colormap': {'A': 'red', 'B': 'green', 'C': 'blue'},
        'colormap': 'gray',
    }

    text_manager = TextManager(
        string='class', properties=properties, color=color
    )

    # ContinuousColorEncoding is the first in the ColorEncoding
    # and can be instantiated with a subset of the dictionary's
    # entries, so is instantiated without an error or warning.
    # To change this behavior to error, update the model config
    # with `extra = 'forbid'`.
    color_array = text_manager.color._get_array(properties, 3)
    expected_color_array = transform_color([[0.5] * 3, [0] * 3, [1] * 3])
    _assert_colors_equal(color_array, expected_color_array)


def test_from_layer():
    text = {
        'string': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([1, 0.5, 0]),
    }

    text_manager = TextManager._from_layer(
        text=text,
        properties=properties,
    )

    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_array_equal(string_array, ['A', 'B', 'C'])
    np.testing.assert_array_equal(text_manager.translation, [-0.5, 1])
    assert not text_manager.visible


def test_update_from_layer():
    text = {
        'string': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([1, 0.5, 0]),
    }
    text_manager = TextManager._from_layer(
        text=text,
        properties=properties,
    )

    text = {
        'string': 'Conf: {confidence:.2f}',
        'translation': [1.5, -2],
        'size': 9000,
    }
    text_manager._update_from_layer(text=text, properties=properties)

    string_array = text_manager.string._get_array(properties, 3)
    np.testing.assert_array_equal(
        string_array, ['Conf: 1.00', 'Conf: 0.50', 'Conf: 0.00']
    )
    np.testing.assert_array_equal(text_manager.translation, [1.5, -2])
    assert text_manager.visible
    assert text_manager.size == 9000


def test_update_from_layer_with_invalid_value_fails_safely():
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([1, 0.5, 0]),
    }
    text_manager = TextManager._from_layer(
        text='class',
        properties=properties,
    )
    before = text_manager.copy(deep=True)

    text = {
        'string': 'confidence',
        'size': -3,
    }

    with pytest.raises(ValidationError):
        text_manager._update_from_layer(text=text, properties=properties)

    assert text_manager == before


def test_update_from_layer_with_warning_only_one_emitted():
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager._from_layer(
        text='class',
        properties=properties,
    )

    text = {
        'string': 'class',
        'blending': 'opaque',
    }

    with pytest.warns(RuntimeWarning) as record:
        text_manager._update_from_layer(text=text, properties=properties)

    assert len(record) == 1


def _assert_colors_equal(actual, expected):
    actual_array = transform_color(actual)
    expected_array = transform_color(expected)
    np.testing.assert_allclose(actual_array, expected_array)
