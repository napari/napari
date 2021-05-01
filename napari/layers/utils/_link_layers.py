from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from itertools import combinations, permutations, product
from typing import TYPE_CHECKING, Callable, Iterable, Tuple

if TYPE_CHECKING:
    from napari.layers import Layer

from ...utils.translations import trans

#: Record of already linked layers... to avoid duplicating callbacks
#  in the form of {(id(layer1), id(layer2), attribute_name) -> callback}
LinkKey = Tuple[int, int, str]
Unlinker = Callable[[], None]
_LINKED_LAYERS: dict[LinkKey, Unlinker] = dict()


def link_layers(
    layers: Iterable['Layer'], attributes: Iterable[str] = ()
) -> list[LinkKey]:
    """Link ``attributes`` between all layers in ``layers``.

    This essentially performs the following operation:

    .. code-block:: python

       for lay1, lay2 in permutations(layers, 2):
           for attr in attributes:
               lay1.events.<attr>.connect(_set_lay2_<attr>)

    Recursion is prevented by checking for value equality prior to setting.

    Parameters
    ----------
    layers : Iterable[napari.layers.Layer]
        The set of layers to link
    attributes : Iterable[str], optional
        The set of attributes to link.  If not provided (the default),
        *all*, event-providing attributes that are common to all ``layers``
        will be linked.

    Returns
    -------
    links: list of (int, int, str) keys
        The links created during execution of the function. The first two
        elements of each tuple are the ids of the two layers, and the last
        element is the linked attribute.

    Raises
    ------
    ValueError
        If any of the attributes provided are not valid "event-emitting"
        attributes, or are not shared by all of the layers provided.

    Examples
    --------
    >>> data = np.random.rand(3, 64, 64)
    >>> viewer = napari.view_image(data, channel_axis=0)
    >>> link_layers(viewer.layers)  # doctest: +SKIP
    """

    from ...utils.misc import pick_equality_operator

    valid_attrs = _get_common_evented_attributes(layers)

    # now, ensure that the attributes requested are valid
    attr_set = set(attributes)
    if attributes:
        extra = attr_set - valid_attrs
        if extra:
            raise ValueError(
                trans._(
                    "Cannot link attributes that are not shared by all layers: {extra}. Allowable attrs include:\n{valid_attrs}",
                    deferred=True,
                    extra=extra,
                    valid_attrs=valid_attrs,
                )
            )
    else:
        # if no attributes are specified, ALL valid attributes are linked.
        attr_set = valid_attrs

    # now, connect requested attributes between all requested layers.
    links = []
    for (lay1, lay2), attribute in product(permutations(layers, 2), attr_set):

        key = _link_key(lay1, lay2, attribute)
        # if the layers and attribute are already linked then ignore
        if key in _LINKED_LAYERS:
            continue

        def _make_l2_setter(l1=lay1, l2=lay2, attr=attribute):
            # get a suitable equality operator for this attribute type
            eq_op = pick_equality_operator(getattr(l1, attr))

            def setter(event=None):
                new_val = getattr(l1, attr)
                # this line is the important part for avoiding recursion
                if not eq_op(getattr(l2, attr), new_val):
                    setattr(l2, attr, new_val)

            setter.__doc__ = f"Set {attr!r} on {l1} to that of {l2}"
            setter.__qualname__ = f"set_{attr}_on_layer_{id(l2)}"
            return setter

        # acually make the connection
        callback = _make_l2_setter()
        emitter_group = getattr(lay1.events, attribute)
        emitter_group.connect(callback)

        # store the connection so that we don't make it again.
        # and save an "unlink" function for the key.
        _LINKED_LAYERS[key] = partial(emitter_group.disconnect, callback)
        links.append(key)
    return links


def unlink_layers(layers: Iterable['Layer'], attributes: Iterable[str] = ()):
    """Unlink previously linked ``attributes`` between all layers in ``layers``.

    Parameters
    ----------
    layers : Iterable[napari.layers.Layer]
        The list of layers to unlink.  All combinations of layers provided will
        be unlinked.  If a single layer is provided, it will be unlinked from
        all other layers.
    attributes : Iterable[str], optional
        The set of attributes to unlink.  If not provided, all connections
        between the provided layers will be unlinked.
    """
    layer_ids = [id(layer) for layer in layers]
    if not layer_ids:
        raise ValueError(
            trans._(
                "Must provide at least one layer to unlink",
                deferred=True,
            )
        )
    elif len(layer_ids) == 1:
        layer_id = layer_ids[0]
        keys = (k for k in list(_LINKED_LAYERS) if layer_id in k[:2])
    else:
        layer_combos = {frozenset(i) for i in combinations(layer_ids, 2)}
        keys = (k for k in list(_LINKED_LAYERS) if set(k[:2]) in layer_combos)
    if attributes:
        keys = (k for k in keys if k[2] in attributes)
    _unlink_keys(keys)


@contextmanager
def layers_linked(layers: Iterable['Layer'], attributes: Iterable[str] = ()):
    """Context manager that temporarily links ``attributes`` on ``layers``."""
    links = link_layers(layers, attributes)
    try:
        yield
    finally:
        _unlink_keys(links)


def _get_common_evented_attributes(
    layers: Iterable['Layer'],
    exclude: set[str] = {'thumbnail', 'status', 'name', 'data'},
    with_private=False,
) -> set[str]:
    """Get the set of common, non-private evented attributes in ``layers``.

    Not all layer events are attributes, and not all attributes have
    corresponding events.  Here we get the set of valid, non-private attributes
    that are both events and attributes for the provided layer set.

    Parameters
    ----------
    layers : iterable
        A set of layers to evaluate for attribute linking.
    exclude : set, optional
        Layer attributes that make no sense to link, or may error on changing.
        {'thumbnail', 'status', 'name', 'data'}
    with_private : bool, optional
        include private attributes

    Returns
    -------
    names : set of str
        A set of attribute names that may be linked between ``layers``.
    """
    from inspect import ismethod

    try:
        first_layer = next(iter(layers))
    except StopIteration:
        raise ValueError(
            trans._(
                "``layers`` iterable must have at least one layer",
                deferred=True,
            )
        )

    common_events = set.intersection(*(set(lay.events) for lay in layers))
    common_attrs = set.intersection(*(set(dir(lay)) for lay in layers))
    if not with_private:
        common_attrs = {x for x in common_attrs if not x.startswith("_")}
    common = common_events & common_attrs - exclude

    # lastly, discard any method-only events (we just want attrs)
    for attr in set(common_attrs):
        # properties do not count as methods and will not be excluded
        if ismethod(getattr(first_layer.__class__, attr, None)):
            common.discard(attr)

    return common


def _link_key(lay1: 'Layer', lay2: 'Layer', attr: str) -> LinkKey:
    """Generate a "link key" for these layers and attribute."""
    return (id(lay1), id(lay2), attr)


def _unlink_keys(keys: Iterable[LinkKey]):
    """Disconnect layer linkages by keys."""
    for key in keys:
        disconnecter = _LINKED_LAYERS.pop(key, None)
        if disconnecter:
            disconnecter()
