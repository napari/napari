import numpy as np
import pytest

from napari.layers.utils.text_manager import TextManager
from napari.utils.colormaps.standardize_color import transform_color


def test_empty_text_manager_property():
    """Test creating an empty text manager in property mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(text='confidence', properties=properties)
    assert text_manager.values.size == 0

    # add a text element
    properties['confidence'] = np.array([0.5])
    text_manager.add(1)
    np.testing.assert_equal(text_manager.values, ['0.5'])


def test_empty_text_manager_format():
    """Test creating an empty text manager in formatted mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text = 'confidence: {confidence:.2f}'
    text_manager = TextManager(text=text, properties=properties)
    assert text_manager.values.size == 0

    # add a text element
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    np.testing.assert_equal(text_manager.values, ['confidence: 0.50'])


def test_text_manager_property():
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, properties=properties)
    np.testing.assert_equal(text_manager.values, classes)

    # add new text with properties
    properties['class'] = np.concatenate((classes, ['A']))
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    np.testing.assert_equal(text_manager.values, properties['class'])

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.values, properties['class'][1::])


def test_text_manager_format():
    text = 'confidence: {confidence:.2f}'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    expected_text = np.array(
        ['confidence: 0.50', 'confidence: 0.30', 'confidence: 1.00']
    )
    text_manager = TextManager(text=text, properties=properties)
    np.testing.assert_equal(text_manager.values, expected_text)

    # add new text with properties
    properties['class'] = np.concatenate((classes, ['A']))
    properties['confidence'] = np.concatenate(
        (properties['confidence'], [0.5])
    )
    text_manager.add(1)
    expected_text_2 = np.concatenate([expected_text, ['confidence: 0.50']])
    np.testing.assert_equal(text_manager.values, expected_text_2)

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
    np.testing.assert_equal(text_manager.values, expected_text_2[1::])


def test_refresh_text():
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, properties=properties)

    new_classes = np.array(['D', 'E', 'F'])
    new_properties = {
        'class': new_classes,
        'confidence': np.array([0.5, 0.3, 1]),
    }
    text_manager.refresh_text(new_properties)
    np.testing.assert_equal(new_classes, text_manager.values)


def test_equality():
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager_1 = TextManager(text=text, properties=properties, color='red')
    text_manager_2 = TextManager(text=text, properties=properties, color='red')

    assert text_manager_1 == text_manager_2
    assert not (text_manager_1 != text_manager_2)

    text_manager_3 = TextManager(
        text=text, properties=properties, color='blue'
    )
    assert text_manager_1 != text_manager_3
    assert not (text_manager_1 == text_manager_3)


def test_blending_modes():
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(
        text=text,
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


def test_constant():
    text_manager = TextManager(text='point', properties={})
    assert len(text_manager.values) == 0


def test_constant_add():
    text_manager = TextManager(text='point', properties={})
    text_manager.add(2)
    np.testing.assert_equal(text_manager.values, ['point', 'point'])


def test_constant_remove():
    text_manager = TextManager(text='point', properties={})
    text_manager.add(5)

    text_manager.remove([1, 3])

    np.testing.assert_equal(text_manager.values, ['point', 'point', 'point'])


def test_direct():
    values = ['one', 'two', 'three']
    text_manager = TextManager(text=values, properties={})
    np.testing.assert_array_equal(text_manager.values, values)


def test_direct_add():
    values = ['one', 'two', 'three']
    text_manager = TextManager(text=values, properties={})

    text_manager.add(2)

    np.testing.assert_array_equal(
        text_manager.values, ['one', 'two', 'three', '', '']
    )


def test_direct_remove():
    values = ['one', 'two', 'three', 'four']
    text_manager = TextManager(text=values, properties={})

    text_manager.remove([1, 3])

    np.testing.assert_array_equal(text_manager.values, ['one', 'three'])


def test_multi_color_direct():
    classes = np.array(['A', 'B', 'C'])
    colors = ['red', 'green', 'blue']
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(
        text='class', properties=properties, color=colors
    )

    np.testing.assert_array_equal(
        text_manager.color_values, transform_color(colors)
    )


def test_multi_color_property():
    colors = ['red', 'green', 'blue']
    properties = {'class': colors, 'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(
        text='class', properties=properties, color='class'
    )

    np.testing.assert_array_equal(
        text_manager.color_values, transform_color(colors)
    )


def test_multi_color_property_discrete_map():
    properties = {
        'class': ['A', 'B', 'C'],
        'confidence': np.array([0.5, 0.3, 1]),
    }
    color = {
        'property_name': 'class',
        'mapping': {'A': 'red', 'B': 'green', 'C': 'blue'},
    }

    text_manager = TextManager(
        text='class', properties=properties, color=color
    )

    np.testing.assert_array_equal(
        text_manager.color_values, transform_color(['red', 'green', 'blue'])
    )


def test_multi_color_property_continuous_map():
    properties = {
        'class': ['A', 'B', 'C'],
        'confidence': np.array([0.5, 0, 1]),
    }
    color = {
        'property_name': 'confidence',
        'colormap': 'gray',
    }

    text_manager = TextManager(
        text='class', properties=properties, color=color
    )

    np.testing.assert_allclose(
        text_manager.color_values,
        transform_color([[0.5] * 3, [0] * 3, [1] * 3]),
    )
