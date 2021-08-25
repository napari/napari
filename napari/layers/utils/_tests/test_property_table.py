import numpy as np
import pandas as pd

from napari.layers.utils.property_table import PropertyTable


def test_empty_table():
    property_table = PropertyTable()
    assert property_table.num_properties == 0
    assert property_table.num_values == 0


def test_from_layer_kwargs_with_num_data():
    property_table = PropertyTable.from_layer_kwargs(
        properties=None, num_data=5
    )
    assert property_table.num_properties == 0
    assert property_table.num_values == 5


def test_from_layer_kwargs_with_properties():
    properties = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }

    property_table = PropertyTable.from_layer_kwargs(
        properties=properties, num_data=4
    )

    assert property_table.num_properties == 2
    assert property_table.num_values == 4
    np.testing.assert_array_equal(
        property_table.data['class'], properties['class']
    )
    np.testing.assert_array_equal(
        property_table.data['confidence'], properties['confidence']
    )
    assert property_table.default_values['class'] == 'person'
    assert property_table.default_values['confidence'] == 0.8
    assert len(property_table.choices) == 0


def test_from_layer_kwargs_with_properties_and_choices():
    properties = {
        'class': ['sky', 'person', 'building', 'person'],
    }
    property_choices = {
        'class': ['building', 'person', 'sky'],
    }

    property_table = PropertyTable.from_layer_kwargs(
        properties=properties, property_choices=property_choices, num_data=4
    )

    assert property_table.num_properties == 1
    assert property_table.num_values == 4
    column = property_table.data['class']
    assert column.name == 'class'
    np.testing.assert_array_equal(column, properties['class'])
    assert property_table.default_values['class'] == 'person'
    np.testing.assert_array_equal(
        property_table.choices['class'], property_choices['class']
    )


def test_from_layer_kwargs_with_choices():
    property_choices = {
        'class': ['building', 'person', 'sky'],
    }

    property_table = PropertyTable.from_layer_kwargs(
        property_choices=property_choices, num_data=0
    )

    assert property_table.num_properties == 1
    assert property_table.num_values == 0
    column = property_table.data['class']
    assert column.name == 'class'
    assert len(column) == 0
    assert property_table.default_values['class'] == 'building'
    np.testing.assert_array_equal(
        property_table.choices['class'], property_choices['class']
    )


def test_from_layer_kwargs_with_empty_properties_and_choices():
    properties = {
        'class': [],
    }
    property_choices = {
        'class': ['building', 'person', 'sky'],
    }

    property_table = PropertyTable.from_layer_kwargs(
        properties=properties, property_choices=property_choices, num_data=0
    )

    assert property_table.num_properties == 1
    assert property_table.num_values == 0
    column = property_table.data['class']
    assert column.name == 'class'
    assert len(column) == 0
    assert property_table.default_values['class'] == 'building'
    np.testing.assert_array_equal(
        property_table.choices['class'], property_choices['class']
    )


def test_from_layer_kwargs_with_properties_as_dataframe():
    properties = pd.DataFrame(
        {
            'class': pd.Series(
                ['sky', 'person', 'building', 'person'],
                dtype=pd.CategoricalDtype(
                    categories=('building', 'person', 'sky')
                ),
            ),
            'confidence': pd.Series([0.2, 0.5, 1, 0.8]),
        }
    )

    property_table = PropertyTable.from_layer_kwargs(properties=properties)

    np.testing.assert_array_equal(property_table.data, properties)
    assert property_table.default_values['class'] == 'person'
    assert property_table.default_values['confidence'] == 0.8


def test_resize_smaller():
    properties = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }
    property_table = PropertyTable(properties)

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
    properties = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }
    property_table = PropertyTable(properties)

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


def test_remove():
    properties = {
        'class': ['sky', 'person', 'building', 'person'],
        'confidence': [0.2, 0.5, 1, 0.8],
    }
    property_table = PropertyTable(properties)

    property_table.remove([1, 3])

    assert property_table.num_values == 2
    np.testing.assert_array_equal(
        property_table.data['class'],
        ['sky', 'building'],
    )
    np.testing.assert_array_equal(
        property_table.data['confidence'],
        [0.2, 1],
    )
