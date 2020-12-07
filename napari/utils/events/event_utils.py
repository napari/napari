def disconnect_events(source, listener):
    """Disconnect all events between a source and a listener.

    Parameters
    ----------
    source : Object
        Any object with an event emitter.
    listener: Object
        Any object that has been connected too.
    """
    for em in source.events.emitters.values():
        for callback in em.callbacks:
            # Callback is a tuple of a weak reference and method name
            if (
                isinstance(callback, tuple)
                and callback[0] is listener.__weakref__
            ):
                em.disconnect(callback)
