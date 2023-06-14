import pytest

from napari._qt.widgets.qt_plugin_sorter import QtPluginSorter, rst2html


@pytest.mark.parametrize(
    'text,expected_text',
    [
        ("", ""),
        (
            """Return a function capable of loading ``path`` into napari, or ``None``.

    This is the primary "**reader plugin**" function.  It accepts a path or
    list of paths, and returns a list of data to be added to the ``Viewer``.
    The function may return ``[(None, )]`` to indicate that the file was read
    successfully, but did not contain any data.

    The main place this hook is used is in :func:`Viewer.open()
    <napari.components.viewer_model.ViewerModel.open>`, via the
    :func:`~napari.plugins.io.read_data_with_plugins` function.

    It will also be called on ``File -> Open...`` or when a user drops a file
    or folder onto the viewer. This function must execute **quickly**, and
    should return ``None`` if the filepath is of an unrecognized format for
    this reader plugin.  If ``path`` is determined to be recognized format,
    this function should return a *new* function that accepts the same filepath
    (or list of paths), and returns a list of ``LayerData`` tuples, where each
    tuple is a 1-, 2-, or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data,
    meta, layer_type)``.

    ``napari`` will then use each tuple in the returned list to generate a new
    layer in the viewer using the :func:`Viewer._add_layer_from_data()
    <napari.components.viewer_model.ViewerModel._add_layer_from_data>`
    method.  The first, (optional) second, and (optional) third items in each
    tuple in the returned layer_data list, therefore correspond to the
    ``data``, ``meta``, and ``layer_type`` arguments of the
    :func:`Viewer._add_layer_from_data()
    <napari.components.viewer_model.ViewerModel._add_layer_from_data>`
    method, respectively.

    .. important::

       ``path`` may be either a ``str`` or a ``list`` of ``str``.  If a
       ``list``, then each path in the list can be assumed to be one part of a
       larger multi-dimensional stack (for instance: a list of 2D image files
       that should be stacked along a third axis). Implementations should do
       their own checking for ``list`` or ``str``, and handle each case as
       desired.""",
            'Return a function capable of loading <code>path</code> into napari, or <code>None</code>.<br><br>    '
            'This is the primary "<strong>reader plugin</strong>" function.  It accepts a path or<br>    '
            'list of paths, and returns a list of data to be added to the <code>Viewer</code>.<br>    '
            'The function may return <code>[(None, )]</code> to indicate that the file was read<br>    '
            'successfully, but did not contain any data.<br><br>    '
            'The main place this hook is used is in <code>Viewer.open()</code>, via the<br>    '
            '<code>read_data_with_plugins</code> function.<br><br>    '
            'It will also be called on <code>File -> Open...</code> or when a user drops a file<br>    '
            'or folder onto the viewer. This function must execute <strong>quickly</strong>, and<br>    '
            'should return <code>None</code> if the filepath is of an unrecognized format for<br>    '
            'this reader plugin.  If <code>path</code> is determined to be recognized format,<br>    '
            'this function should return a <em>new</em> function that accepts the same filepath<br>    '
            '(or list of paths), and returns a list of <code>LayerData</code> tuples, where each<br>    '
            'tuple is a 1-, 2-, or 3-tuple of <code>(data,)</code>, <code>(data, meta)</code>, or <code>(data,<br>    '
            'meta, layer_type)</code>.<br><br>    <code>napari</code> will then use each tuple in the returned list to generate a new<br>    '
            'layer in the viewer using the <code>Viewer._add_layer_from_data()</code><br>    '
            'method.  The first, (optional) second, and (optional) third items in each<br>    '
            'tuple in the returned layer_data list, therefore correspond to the<br>    '
            '<code>data</code>, <code>meta</code>, and <code>layer_type</code> arguments of the<br>    '
            '<code>Viewer._add_layer_from_data()</code><br>    method, respectively.<br><br>    .. important::<br><br>'
            '       <code>path</code> may be either a <code>str</code> or a <code>list</code> of <code>str</code>.  If a<br>'
            '       <code>list</code>, then each path in the list can be assumed to be one part of a<br>       '
            'larger multi-dimensional stack (for instance: a list of 2D image files<br>       '
            'that should be stacked along a third axis). Implementations should do<br>       '
            'their own checking for <code>list</code> or <code>str</code>, and handle each case as<br>       '
            'desired.',
        ),
    ],
)
def test_rst2html(text, expected_text):
    assert rst2html(text) == expected_text


def test_create_qt_plugin_sorter(qtbot):
    plugin_sorter = QtPluginSorter()
    qtbot.addWidget(plugin_sorter)

    # Check initial hook combobox items
    hook_combo_box = plugin_sorter.hook_combo_box
    combobox_items = [
        hook_combo_box.itemText(idx) for idx in range(hook_combo_box.count())
    ]
    assert combobox_items == [
        'select hook... ',
        'get_reader',
        'get_writer',
        'write_image',
        'write_labels',
        'write_points',
        'write_shapes',
        'write_surface',
        'write_vectors',
    ]


@pytest.mark.parametrize(
    "hook_name,help_info",
    [
        ('select hook... ', ''),
        (
            'get_reader',
            'This is the primary "<strong>reader plugin</strong>" function.  It accepts a path or<br>    list of paths, and returns a list of data to be added to the <code>Viewer</code>.<br>',
        ),
        (
            'get_writer',
            'This function will be called whenever the user attempts to save multiple<br>    layers (e.g. via <code>File -> Save Layers</code>, or<br>    <code>save_layers</code>).<br>',
        ),
        (
            'write_labels',
            'It is the responsibility of the implementation to check any extension on<br>    <code>path</code> and return <code>None</code> if it is an unsupported extension.',
        ),
        (
            'write_points',
            'It is the responsibility of the implementation to check any extension on<br>    <code>path</code> and return <code>None</code> if it is an unsupported extension.',
        ),
        (
            'write_shapes',
            'It is the responsibility of the implementation to check any extension on<br>    <code>path</code> and return <code>None</code> if it is an unsupported extension.',
        ),
        (
            'write_surface',
            'It is the responsibility of the implementation to check any extension on<br>    <code>path</code> and return <code>None</code> if it is an unsupported extension.',
        ),
        (
            'write_vectors',
            'It is the responsibility of the implementation to check any extension on<br>    <code>path</code> and return <code>None</code> if it is an unsupported extension.',
        ),
    ],
)
def test_qt_plugin_sorter_help_info(qtbot, hook_name, help_info):
    plugin_sorter = QtPluginSorter()
    qtbot.addWidget(plugin_sorter)

    # Check hook combobox items help tooltip in the info widget
    info_widget = plugin_sorter.info
    hook_combo_box = plugin_sorter.hook_combo_box
    hook_combo_box.setCurrentText(hook_name)

    assert help_info in info_widget.toolTip()
