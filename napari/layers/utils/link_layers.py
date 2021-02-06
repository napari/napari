from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from napari.layers import Layer

#: Record of already linked layers... to avoid duplicating callbacks
#  in the form of (id(layer1), id(layer2), attribute_name)
_LINKED: set[tuple[int, int, str]] = set()


# attributes that make no sense to link, or error on changing
# we may want to include colormap here, but in most cases, it works fine.
PROHIBITED = {'thumbnail', 'status', 'name'}


def _get_common_evented_attributes(
    layers: Iterable['Layer'],
    prohibited: set[str] = PROHIBITED,
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
    prohibited : set, optional
        A set of attributes to exclude, by default PROHIBITED
    private : bool, optional
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
        raise ValueError("``layers`` iterable must have at least one layer")
    common_events = set.intersection(*(set(lay.events) for lay in layers))
    common_attrs = set.intersection(*(set(dir(lay)) for lay in layers))
    if not with_private:
        common_attrs = {x for x in common_attrs if not x.startswith("_")}
    common = common_events & common_attrs - prohibited

    # lastly, discard any method-only events (we just want attrs)
    for attr in set(common_attrs):
        # properties do not count as methods and will not be excluded
        if ismethod(getattr(first_layer.__class__, attr, None)):
            common.discard(attr)

    return common


def experimental_link_layers(
    layers: Iterable['Layer'], attributes: Iterable[str] = ()
):
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

    Raises
    ------
    ValueError
        If any of the attributes provided are not valid "event-emitting"
        attributes, or are not shared by all of the layers provided.

    Examples
    --------
    >>> data = np.random.rand(3, 64, 64)
    >>> viewer = napari.view_image(data, channel_axis=0)
    >>> experimental_link_layers(viewer.layers)
    """
    from itertools import permutations, product

    from ...utils.misc import get_equality_operator

    valid_attrs = _get_common_evented_attributes(layers)

    # now, ensure that the attributes requested are valid
    attr_set = set(attributes)
    if attributes:
        extra = attr_set - valid_attrs
        if extra:
            raise ValueError(
                "Cannot link attributes that are not shared by all "
                f"layers: {extra}. Allowable attrs include:\n{valid_attrs}"
            )
    else:
        # if no attributes are specified, ALL valid attributes are linked.
        attr_set = valid_attrs

    # now, connect requested attributes between all requested layers.
    for (lay1, lay2), attr in product(permutations(layers, 2), attr_set):

        key = (id(lay1), id(lay2), attr)
        if key in _LINKED:
            continue

        def _make_l2_setter(eq_op, l1=lay1, l2=lay2, attr=attr):
            # where `eq_op` is an "equality checking function" that is suitable
            # for the attribute object type.  (see ``get_equality_operator``)

            def setter(event=None):
                new_val = getattr(l1, attr)
                # this line is the important part for avoiding recursion
                if not eq_op(getattr(l2, attr), new_val):
                    setattr(l2, attr, new_val)

            setter.__doc__ = f"Set {attr!r} on {lay2} to that of {lay1}"
            setter.__qualname__ = f"set_{attr}_on_layer_{id(lay2)}"
            return setter

        # get a suitable equality operator for this attribute type
        eq_op = get_equality_operator(getattr(lay1, attr))
        # acually make the connection
        getattr(lay1.events, attr).connect(_make_l2_setter(eq_op))
        # store the connection so that we don't make it again.
        _LINKED.add(key)
