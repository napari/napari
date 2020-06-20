import numpy as np
import pytest

from napari.layers.utils.text import TextManager


@pytest.mark.parametrize(
    'text', ['repeated_text', np.repeat('repeated_text', 3)]
)
def test_text_manager_direct(text):
    """Test initializing a TextManager directly setting the text"""
    n_text = 3
    color = 'red'
    text_manager = TextManager(text=text, n_text=n_text, color=color)
    expected_text = np.repeat('repeated_text', 3)
    np.testing.assert_equal(text_manager.text, expected_text)
    np.testing.assert_allclose(text_manager.color, [1, 0, 0, 1])
    assert text_manager.mode == 'direct'

    # add text elements with repeat
    text_manager.add('hello', n_text=3)
    expected_text_2 = np.concatenate([expected_text, np.repeat('hello', 3)])
    np.testing.assert_equal(text_manager.text, expected_text_2)

    # add text element with array
    new_text = ['bonjour', 'hola']
    text_manager.add(new_text, 2)
    expected_text_3 = np.concatenate([expected_text_2, new_text])
    np.testing.assert_equal(text_manager.text, expected_text_3)


def test_text_manager_property():
    n_text = 3
    text = 'class'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.text, classes)
    assert text_manager.mode == 'property'

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    expected_text_2 = np.concatenate([classes, ['A']])
    np.testing.assert_equal(text_manager.text, expected_text_2)


def test_text_manager_format():
    n_text = 3
    text = 'confidence: {confidence:.2f}'
    classes = np.array(['A', 'B', 'C'])
    properties = {'class': classes, 'confidence': np.array([0.5, 0.3, 1])}
    expected_text = np.array(
        ['confidence: 0.50', 'confidence: 0.30', 'confidence: 1.00']
    )
    text_manager = TextManager(text=text, n_text=n_text, properties=properties)
    np.testing.assert_equal(text_manager.text, expected_text)
    assert text_manager.mode == 'formatted'

    # add new text with properties
    new_properties = {'class': np.array(['A']), 'confidence': np.array([0.5])}
    text_manager.add(new_properties, 1)
    expected_text_2 = np.concatenate([expected_text, ['confidence: 0.50']])
    np.testing.assert_equal(text_manager.text, expected_text_2)
