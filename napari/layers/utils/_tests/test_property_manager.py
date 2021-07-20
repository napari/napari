import numpy as np

from napari.layers.utils.property_manager import Property, PropertyManager


def test_property_manager_empty():
    manager = PropertyManager()
    assert len(manager._properties) == 0


# def test_property_manager_from_property_arrays_plain():
#    property_arrays = {
#            'class': ['sky', 'person', 'building', 'person'],
#            'confidence': [0.2, 0.5, 1, 0.8],
#    }
#
#    manager = PropertyManager(properties=property_arrays)
#
#    np.testing.assert_array_equal(manager.properties['class'].values, property_arrays['class'])
#    np.testing.assert_array_equal(manager.properties['confidence'].values, property_arrays['confidence'])
#    assert manager.properties['class'].default_value == 'person'
#    assert manager.properties['confidence'].default_value == 0.8


def test_property_manager_from_property_list():
    property_list = [
        Property.from_values('class', ['sky', 'person', 'building', 'person']),
        Property.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]

    manager = PropertyManager.from_property_list(property_list)

    assert manager._properties['class'] == property_list[0]
    assert manager._properties['confidence'] == property_list[1]


def test_property_manager_from_property_arrays():
    property_arrays = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }

    manager = PropertyManager.from_property_arrays(property_arrays)

    np.testing.assert_array_equal(
        manager._properties['class'].values, property_arrays['class']
    )
    np.testing.assert_array_equal(
        manager._properties['confidence'].values, property_arrays['confidence']
    )
    assert manager._properties['class'].default_value == 'person'
    assert manager._properties['confidence'].default_value == 0.8


def test_property_manager_from_property_choices():
    property_choices = {
        'class': ['building', 'person', 'sky'],
        # TODO: allow choices to be None or define more fancy typing for real numbers.
        'confidence': [0.2, 0.5, 0.8, 1],
    }

    manager = PropertyManager.from_property_choices(property_choices)

    np.testing.assert_array_equal(
        manager._properties['class'].choices, property_choices['class']
    )
    np.testing.assert_array_equal(
        manager._properties['confidence'].choices,
        property_choices['confidence'],
    )
    assert manager._properties['class'].default_value == 'building'
    assert manager._properties['confidence'].default_value == 0.2


def test_resize_smaller():
    property_list = [
        Property.from_values('class', ['sky', 'person', 'building', 'person']),
        Property.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]
    manager = PropertyManager.from_property_list(property_list)

    manager.resize(2)

    np.testing.assert_array_equal(
        manager._properties['class'].values, ['sky', 'person']
    )
    np.testing.assert_array_equal(
        manager._properties['confidence'].values, [0.2, 0.5]
    )


def test_resize_larger():
    property_list = [
        Property.from_values('class', ['sky', 'person', 'building', 'person']),
        Property.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]
    manager = PropertyManager.from_property_list(property_list)

    manager.resize(6)

    np.testing.assert_array_equal(
        manager._properties['class'].values,
        ['sky', 'person', 'building', 'person', 'person', 'person'],
    )
    np.testing.assert_array_equal(
        manager._properties['confidence'].values, [0.2, 0.5, 1, 0.8, 0.8, 0.8]
    )
