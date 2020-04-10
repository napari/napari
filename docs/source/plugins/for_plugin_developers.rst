.. _plugins-for-plugin-developers:

.. role:: python(code)
   :language: python

creating a napari plugin
========================

This document explains how to extend napari functionality by writing a plugin
that can be installed with ``pip`` and autodetected by napari.  For in-depth
information on how plugins are used internally in napari, see
:ref:`plugins-for-napari-developers`.


Overview
--------

``napari`` allows developers to extend the functionality of the program as
follows:

1. **Hooks**: We (napari) identify specific places ("*hooks*") in the napari
codebase where we would like external developers to be able to extend
functionality. For example, when a user tries to open a filepath in napari, we
might want to enable plugins to extend the file formats that can be handled.  A
*hook*, then, is the place within napari where we
"promise" to call functions created by external developers.

2. **Hook Specifications**:  We then create "*hook specifications*", which are
well-documented function signatures that define the API (or
"contract") that a plugin developer must adhere to when writing their function
that we promise to call somewhere in the napari codebase.

3. **Hook Implementations**: Plugin developers then write functions ("*hook
implementations*") and mark that function as meeting the requirements of a
specific *hook specification* offered by napari (using a decorator as
:ref:`described below <hookimpl-decorator>`).

4. **Plugin discovery**: Using :ref:`one of two methods <hookimpl-decorator>`,
plugins that are installed in the same python environment as napari can make
themselves known to napari. ``napari`` will then scan plugin modules for *hook
implementations* that will then be called at the appropriate time place during
the execution of ``napari``.


Step 1: Choose a hook specification to implement
------------------------------------------------

The functionality of plugins, as currently designed and implemented in
``napari``, is rather specific in scope: They are *not* just independent code
blocks with their own GUIs that show up next to the main napari window. Rather,
plugin developers must decide which of the current *hook specifications*
defined by napari that they would like to implement.

For a complete list of *hook specifications* that developers can implement, see
the :ref:`hook-specifications-reference`.

A single plugin package may implement more than one *hook specification*, but
may not declare more the one *hook implementation* for any given specification.


.. NOTE::
   One of the primary ways that we will extend the functionality of napari over
   time is by identifying new ideas for *hook specifications* that developers
   can implement.  If you have a plugin idea that requires napari to create a
   new hook specification, we'd love to hear about it!  Please think about what
   the signature of your proposed hook specification would look like, and where
   within the napari codebase you'd like your hook implementation to be called,
   and `open a feature request
   <https://github.com/napari/napari/issues/new?template=feature_request.md>`_
   in the napari issue tracker with your proposal.

Let's take the :func:`~napari.plugins.hook_specifications.napari_get_reader`
hook (our primary "reader plugin" hook) as an example.  It is defined as:

.. code-block:: python

   LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
   ReaderFunction = Callable[[str], List[LayerData]]

   @napari_hook_specification(firstresult=True)
   def napari_get_reader(
       path: Union[str, List[str]]
   ) -> Optional[ReaderFunction]:
       ...

Note that it takes a ``str`` or a ``list`` of ``str`` and either returns
``None`` or a function.  From the :func:`docstring
<napari.plugins.hook_specifications.napari_get_reader>` of the hook
specification, we see that the implementation should return ``None`` if the
path is of an unrecognized format, otherwise it should return a
``ReaderFunction``, which is a function that takes a ``str`` (the filepath to
read) and returns a ``list`` of ``LayerData``, where ``LayerData`` is any one
of ``(data,)``, ``(data, meta)``, or ``(data, meta, layer_type)``.

That seems like a bit of a mouthful!  But it's a precise (though flexible)
contract that you can follow, and know that napari will handle the rest.


Step 2: Write your hook implementation
--------------------------------------

Once you have identified the :ref:`hook specification
<hook-specifications-reference>` that you want to implement, you have to create
a *hook implementation*: a function that accepts the arguments specified by the
hook specification signature and returns a value with the expected return type.

Here's an example hook implementation for
:func:`~napari.plugins.hook_specifications.napari_get_reader` that enables
napari to open a numpy binary file with a ``.npy`` extension (previously saved
with :func:`numpy.save`)

.. code-block:: python

   import numpy as np
   from napari.plugins import napari_hook_implementation


   def npy_file_reader(path):
      array = np.load(path)
      # return it as a list of LayerData tuples,
      # here with no optional metadata
      return [(array,)]


   # this line is explained below in "Decorating your function..."
   @napari_hook_implementation
   def napari_get_reader(path):
      # remember, path can be a list, so we check it's type first...
      # (this example plugin doesn't handle lists)
      if isinstance(path, str) and path.endswith(".npy"):
         # If we recognize the format, we return the actual reader function
         return npy_file_reader
      # otherwise we return None.
      return None


.. _hookimpl-decorator:

Decorating your function with ``pluggy.HookimplMarker``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to let ``napari`` know that one of your functions satisfies the API of
one of the napari *hook specifications*, you must decorate your function with
an instance of `pluggy.HookimplMarker
<https://pluggy.readthedocs.io/en/latest/#marking-hooks>`_, initialized with
the name ``"napari"``.  As a convenience, napari provides this decorator at
``napari.plugins.napari_hook_implementation`` as shown in the example above.

However, it's not required to import from or depend on napari *at all* when
writing a plugin. You can construct your own ``napari_hook_implementation``
decorator importing directly from ``pluggy`` (a very lightweight dependency
that uses only standard lib python).

.. code-block:: python

   from pluggy import HookimplMarker

   napari_hook_implementation = HookimplMarker("napari")

Matching hook implementations to specifications
"""""""""""""""""""""""""""""""""""""""""""""""

Currently (as of April, 2020), the only way that napari knows *which* hook
specification your implementation matches is by looking at the *name* of your
decorated function.  So in the example above, it was **critical** that our hook
implementation was literally named ``napari_get_reader``:


.. code-block:: python

   @napari_hook_implementation
   def napari_get_reader(path: str):
      ...

However, `a pull request has been merged at pluggy
<https://github.com/pytest-dev/pluggy/pull/251>`_ that will enable you to mark
*any* function as satisfying a napari hook specification (regardless of the
function's name) by providing the name of the target hook specification to the
``specname`` argument in your implementation decorator:

.. code-block:: python

   @napari_hook_implementation(specname="napari_get_reader")
   def whatever_name_you_want(path: str):
      ...

(Monitor the `pluggy changelog
<https://github.com/pytest-dev/pluggy/blob/master/CHANGELOG.rst>`_ for release
of PR #251.)

.. _plugin-discovery:

Step 3: Make your plugin discoverable
-------------------------------------

Packages and modules installed in the same environment as ``napari`` may make
themselves "discoverable" to napari using one of two common conventions
outlined in the `Python Packaging Authority guide
<https://packaging.python.org/guides/creating-and-discovering-plugins/>`_.

Using naming convention
^^^^^^^^^^^^^^^^^^^^^^^

``napari`` will look for *hook implementations* (i.e. functions decorated with
the ``HookimplMarker("napari")`` decorator) in all top-level modules in
``sys.path`` that begin with the name ``napari_`` (e.g. "``napari_myplugin``").

One potential benefit of using discovery by naming convention is that it will
allow ``napari`` to query the PyPi API to search for potential plugins.

.. _entry-point-discovery:

Using package metadata
^^^^^^^^^^^^^^^^^^^^^^

By providing an ``entry_points`` argument with the key ``napari.plugin`` to
``setup()`` in ``setup.py``, plugins can register themselves for discovery
(even if their names do not begin with "``napari_``").

For example if you have a package named ``mypackage`` with a submodule
``napari_plugin`` where you have decorated one or more napari hook
implementations, then if you include in ``setup.py``:

.. code-block:: python

   # setup.py

   setup(
      ...
      entry_points={'napari.plugin': 'plugin_name = mypackage.napari_plugin'},
      ...
   )

... then napari will search the ``mypackage.napari_plugin`` module for
functions decorated with the ``HookimplMarker("napari")`` decorator and
register them the plugin name ``"plugin_name"``.

One benefit of using this approach is that if you already have an existing
pip-installable package, you can extend support for ``napari`` without having
to rename your package, simply by identifying the module in your package that
has the hook implementations.

A user would then be able to use ``napari``, extended with your package's
functionality by simply installing your package along with napari:

.. code:: bash

   pip install napari mypackage


Cookiecutter template
---------------------

To quickly generate a new napari plugin project, you may wish to use the
`cookiecutter-napari-plugin
<https://github.com/napari/cookiecutter-napari-plugin>`_ template.  This uses
the `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_ command line
utility, which will ask you a few questions about your project and get you
started with a ready-to-go package layout where you can begin implementing your
plugin.

Install cookiecutter and use the template as follows:

.. code-block:: bash

   pip install cookiecutter
   cookiecutter https://github.com/napari/cookiecutter-napari-plugin


Example Plugins
---------------

For a minimal working plugin example, see the `napari-dv
<https://github.com/tlambert03/napari-dv>`_ plugin, which allows ``napari`` to
read the `Priism/MRC/Deltavision image file format
<https://github.com/tlambert03/mrc>`_.

For a more thorough plugin see `napari-aicsimageio
<https://github.com/AllenCellModeling/napari-aicsimageio>`_, one of the first
community plugins developed for napari.  This plugin takes advantage of
:ref:`entry_point discovery <entry-point-discovery>` to offer multiple
readers for both in-memory and lazy-loading of image files.

Help
----

If you run into trouble creating your plugin, please don't hesitate to reach
out for help in the `Image.sc Forum <https://forum.image.sc/tag/napari>`_.
Alternatively, if you find a bug or have a specific feature request for plugin
support, please open an issue at our `github issue tracker
<https://github.com/napari/napari/issues/new/choose>`_.
