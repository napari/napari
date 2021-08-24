import numpy as np
import pandas as pd

from napari.layers.utils.property_table import PropertyTable


def test_property_table_empty():
    property_table = PropertyTable()
    assert property_table.num_properties == 0
    assert property_table.num_values == 0


def test_property_table_from_layer_kwargs_no_properties_some_data():
    property_table = PropertyTable.from_layer_kwargs(
        properties=None, num_data=5
    )
    assert property_table.num_properties == 0
    assert property_table.num_values == 5


def test_property_table_from_layer_kwargs_values_and_choices_some_data():
    class_values = ['sky', 'person', 'building', 'person']
    class_choices = ['building', 'person', 'sky']
    properties = pd.DataFrame(
        pd.Series(
            name='class',
            data=class_values,
            dtype=pd.CategoricalDtype(categories=class_choices),
        )
    )

    property_table = PropertyTable.from_layer_kwargs(
        properties=properties, num_data=4
    )

    assert property_table.num_properties == 1
    assert property_table.num_values == 4
    column = property_table.data['class']
    assert column.name == 'class'
    np.testing.assert_array_equal(column, class_values)
    assert property_table.default_values['class'] == 'person'
    np.testing.assert_array_equal(
        property_table.choices['class'], class_choices
    )


def test_property_table_from_layer_kwargs_values_and_choices_no_data():
    class_values = []
    class_choices = ['building', 'person', 'sky']
    properties = pd.DataFrame(
        pd.Series(
            name='class',
            data=class_values,
            dtype=pd.CategoricalDtype(categories=class_choices),
        )
    )

    property_table = PropertyTable.from_layer_kwargs(
        properties=properties, num_data=0
    )

    assert property_table.num_properties == 1
    assert property_table.num_values == 0
    column = property_table.data['class']
    assert column.name == 'class'
    np.testing.assert_array_equal(column, class_values)
    # TODO: consider making default value first category value.
    assert property_table.default_values['class'] is None
    np.testing.assert_array_equal(
        property_table.choices['class'], class_choices
    )


def test_property_table_from_property_arrays():
    property_arrays = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }

    property_table = PropertyTable.from_layer_kwargs(
        properties=property_arrays, num_data=4
    )

    assert property_table.num_properties == 2
    assert property_table.num_values == 4
    np.testing.assert_array_equal(
        property_table.data['class'], property_arrays['class']
    )
    np.testing.assert_array_equal(
        property_table.data['confidence'], property_arrays['confidence']
    )
    assert property_table.default_values['class'] == 'person'
    assert property_table.default_values['confidence'] == 0.8


def test_resize_smaller():
    property_arrays = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }
    property_table = PropertyTable(property_arrays)

    property_table.resize(2)

    assert property_table.num_values == 2
    np.testing.assert_array_equal(
        property_table.data['class'], ['sky', 'person']
    )
    np.testing.assert_array_equal(
        property_table.data['confidence'], [0.2, 0.5]
    )
    assert property_table.default_values['class'] == 'person'
    assert property_table.default_values['confidence'] == 0.8


def test_resize_larger():
    property_arrays = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }
    property_table = PropertyTable(property_arrays)

    property_table.resize(6)

    assert property_table.num_values == 6
    np.testing.assert_array_equal(
        property_table.data['class'],
        ['sky', 'person', 'building', 'person', 'person', 'person'],
    )
    np.testing.assert_array_equal(
        property_table.data['confidence'],
        [0.2, 0.5, 1, 0.8, 0.8, 0.8],
    )
