from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from napari.layers import Layer


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

    # not all events are attributes... so start by getting the set of
    # valid, non-private attributes that are both events and attributes
    # for the provided layer set
    common_events = set.intersection(*(set(lay.events) for lay in layers))
    common_attrs = set.intersection(
        *({x for x in dir(lay) if not x.startswith("_")} for lay in layers)
    )
    # attributes that make no sense, or error.
    # we may want to include colormap here, but in most cases, it works fine.
    prohibited = {'thumbnail', 'status', 'name'}
    valid_attrs = common_events & common_attrs - prohibited

    # now, ensure that the attributes requested are valid
    attr_set = set(attributes)
    if attributes:
        extra = attr_set - valid_attrs
        if extra:
            raise ValueError(
                "Attributes provided that are not shared by all "
                f"layers: {extra}"
            )
    else:
        # if no attributes are specified, ALL valid attributes are linked.
        attr_set = valid_attrs

    # now, connect requested attributes between all requested layers.
    for (lay1, lay2), attr in product(permutations(layers, 2), attr_set):

        def _make_l2_setter(eq_op, l1=lay1, l2=lay2, attr=attr):
            # where `eq_op` is an "equality checking function" that is suitable
            # for the attribute object type.  (see ``get_equality_operator``)

            def setter(event=None):
                new_val = getattr(l1, attr)
                # this line is the important part for avoiding recursion
                if not eq_op(getattr(l2, attr), new_val):
                    setattr(l2, attr, new_val)

            setter.__doc__ = f"Set {attr!r} on {lay2} to that of {lay1}"
            return setter

        eq_op = get_equality_operator(getattr(lay1, attr))
        getattr(lay1.events, attr).connect(_make_l2_setter(eq_op))
