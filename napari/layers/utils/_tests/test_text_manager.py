import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils._text_constants import TextMode
from napari.layers.utils.text_manager import TextManager


def test_empty_text_manager_property():
    """Test creating an empty text manager in property mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence', n_text=0, properties=properties
    )
    assert text_manager._mode == TextMode.PROPERTY
    assert text_manager.values.size == 0

    # add a text element
    new_properties = {'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    np.testing.assert_equal(text_manager.values, ['0.5'])


def test_add_many_text_property():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence',
        n_text=0,
        properties=properties,
    )

    text_manager.add({'confidence': np.array([0.5])}, 2)

    np.testing.assert_equal(text_manager.values, ['0.5'] * 2)


def test_empty_text_manager_format():
    """Test creating an empty text manager in formatted mode.
    This is for creating an empty layer with text initialized.
    """
    properties = {'confidence': np.empty(0, dtype=float)}
    text = 'confidence: {confidence:.2f}'
    text_manager = TextManager(text=text, n_text=0, properties=properties)
    assert text_manager._mode == TextMode.FORMATTED
    assert text_manager.values.size == 0

    # add a text element
    new_properties = {'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    np.testing.assert_equal(text_manager.values, ['confidence: 0.50'])


@pytest.mark.xfail(reason='To be fixed with properties refactor.')
def test_add_many_text_formatted():
    properties = {'confidence': np.empty(0, dtype=float)}
    text_manager = TextManager(
        text='confidence: {confidence:.2f}',
        n_text=0,
        properties=properties,
    )

    text_manager.add({'confidence': np.array([0.5])}, 2)

    np.testing.assert_equal(text_manager.values, ['confidence: 0.50'] * 2)


def test_text_manager_property():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.values, classes)
    assert text_manager._mode == TextMode.PROPERTY

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    expected_text_2 = np.concatenate([classes, ['A']])
    np.testing.assert_equal(text_manager.values, expected_text_2)

    # remove the first text element
    text_manager.remove({0})
    np.testing.assert_equal(text_manager.values, expected_text_2[1::])


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
    assert text_manager._mode == TextMode.FORMATTED

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
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


def test_text_with_invalid_format_string_then_constant_text():
    n_text = 3
    text = 'confidence: {confidence:.2f'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(text=text, n_text=n_text, properties=properties)

    np.testing.assert_array_equal(text_manager.values, [text] * n_text)


def test_text_with_format_string_missing_property_then_constant_text():
    n_text = 3
    text = 'score: {score:.2f}'
    properties = {'confidence': np.array([0.5, 0.3, 1])}

    text_manager = TextManager(text=text, n_text=n_text, properties=properties)

    np.testing.assert_array_equal(text_manager.values, [text] * n_text)


def test_text_constant_then_repeat_values():
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}

    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )

    np.testing.assert_array_equal(text_manager.values, ['point'] * n_text)


def test_text_constant_with_no_properties_then_no_values():
    # TODO: we may generate n_text copies as part of the properties refactor.
    text_manager = TextManager(text='point', n_text=3)
    assert len(text_manager.values) == 0


def test_add_with_text_constant_then_ignored():
    # TODO: we may choose not to ignore add as part of the properties refactor.
    n_text = 3
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )
    assert len(text_manager.values) == n_text

    text_manager.add({'class': np.array(['C'])}, 2)

    assert len(text_manager.values) == n_text


def test_add_with_text_constant_init_empty_then_ignored():
    # TODO: we may choose not to ignore add as part of the properties refactor.
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager(text='point', n_text=0, properties=properties)

    text_manager.add({'class': np.array(['C'])}, 2)

    assert len(text_manager.values) == 0


def test_remove_with_text_constant_then_ignored():
    # TODO: we may choose not to ignore remove as part of the properties refactor.
    n_text = 5
    properties = {'class': np.array(['A', 'B', 'C', 'D', 'E'])}
    text_manager = TextManager(
        text='point', n_text=n_text, properties=properties
    )

    text_manager.remove([1, 3])

    np.testing.assert_equal(text_manager.values, ['point'] * n_text)


def test_from_layer():
    text = {
        'text': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([1, 0.5, 0]),
    }

    text_manager = TextManager._from_layer(
        text=text,
        n_text=3,
        properties=properties,
    )

    np.testing.assert_array_equal(text_manager.values, ['A', 'B', 'C'])
    np.testing.assert_array_equal(text_manager.translation, [-0.5, 1])
    assert not text_manager.visible


def test_update_from_layer():
    text = {
        'text': 'class',
        'translation': [-0.5, 1],
        'visible': False,
    }
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'confidence': np.array([1, 0.5, 0]),
    }
    text_manager = TextManager._from_layer(
        text=text,
        n_text=3,
        properties=properties,
    )

    text = {
        'text': 'Conf: {confidence:.2f}',
        'translation': [1.5, -2],
        'size': 9000,
    }
    text_manager._update_from_layer(text=text, n_text=3, properties=properties)

    np.testing.assert_array_equal(
        text_manager.values, ['Conf: 1.00', 'Conf: 0.50', 'Conf: 0.00']
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
        n_text=3,
        properties=properties,
    )
    before = text_manager.copy(deep=True)

    text = {
        'text': 'confidence',
        'size': -3,
    }

    with pytest.raises(ValidationError):
        text_manager._update_from_layer(
            text=text, n_text=3, properties=properties
        )

    assert text_manager == before


def test_update_from_layer_with_warning_only_one_emitted():
    properties = {'class': np.array(['A', 'B', 'C'])}
    text_manager = TextManager._from_layer(
        text='class',
        n_text=3,
        properties=properties,
    )

    text = {
        'text': 'class',
        'blending': 'opaque',
    }

    with pytest.warns(RuntimeWarning) as record:
        text_manager._update_from_layer(
            text=text, n_text=3, properties=properties
        )

    assert len(record) == 1
