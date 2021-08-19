import numpy as np

from napari.layers.utils.property_table import PropertyColumn, PropertyTable


def test_property_table_empty():
    manager = PropertyTable()
    assert len(manager) == 0


def test_property_table_from_property_list():
    property_list = [
        PropertyColumn.from_values(
            'class', ['sky', 'person', 'building', 'person']
        ),
        PropertyColumn.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]

    manager = PropertyTable.from_property_list(property_list)

    assert manager['class'] == property_list[0]
    assert manager['confidence'] == property_list[1]


def test_property_table_from_property_arrays():
    property_arrays = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }

    manager = PropertyTable.from_property_arrays(property_arrays)

    np.testing.assert_array_equal(
        manager['class'].values, property_arrays['class']
    )
    np.testing.assert_array_equal(
        manager['confidence'].values, property_arrays['confidence']
    )
    assert manager['class'].default_value == 'person'
    assert manager['confidence'].default_value == 0.8


def test_property_table_from_property_choices():
    property_choices = {
        'class': ['building', 'person', 'sky'],
        # TODO: allow choices to be None or define more fancy typing for real numbers.
        'confidence': [0.2, 0.5, 0.8, 1],
    }

    manager = PropertyTable.from_property_choices(property_choices)

    np.testing.assert_array_equal(
        manager['class'].choices, property_choices['class']
    )
    np.testing.assert_array_equal(
        manager['confidence'].choices,
        property_choices['confidence'],
    )
    assert manager['class'].default_value == 'building'
    assert manager['confidence'].default_value == 0.2


def test_resize_smaller():
    property_list = [
        PropertyColumn.from_values(
            'class', ['sky', 'person', 'building', 'person']
        ),
        PropertyColumn.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]
    manager = PropertyTable.from_property_list(property_list)

    manager.resize(2)

    np.testing.assert_array_equal(manager['class'].values, ['sky', 'person'])
    np.testing.assert_array_equal(manager['confidence'].values, [0.2, 0.5])


def test_resize_larger():
    property_list = [
        PropertyColumn.from_values(
            'class', ['sky', 'person', 'building', 'person']
        ),
        PropertyColumn.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]
    manager = PropertyTable.from_property_list(property_list)

    manager.resize(6)

    np.testing.assert_array_equal(
        manager['class'].values,
        ['sky', 'person', 'building', 'person', 'person', 'person'],
    )
    np.testing.assert_array_equal(
        manager['confidence'].values, [0.2, 0.5, 1, 0.8, 0.8, 0.8]
    )


def test_property_changed_event():
    property_list = [
        PropertyColumn.from_values(
            'class', ['sky', 'person', 'building', 'person']
        ),
        PropertyColumn.from_values('confidence', [0.2, 0.5, 1, 0.8]),
    ]
    manager = PropertyTable.from_property_list(property_list)
    observed = []
    manager.events.changed.connect(lambda e: observed.append(e))

    new_class_values = PropertyColumn.from_values(
        'class', ['sky', 'person', 'building', 'duck']
    )
    manager['class'] = new_class_values

    assert len(observed) == 1
    assert observed[0].type == 'changed'
    assert observed[0].key == 'class'
    assert observed[0].value == new_class_values
