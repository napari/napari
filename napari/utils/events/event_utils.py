from collections.abc import Iterable


def connect(source, destination, connections=None):
    if connections is None:
        # connection None then do automatically
        for event_name in source.events.emitters.keys():
            event = getattr(source.events, event_name)
            method = getattr(destination, f'_on_{event_name}_change', None)
            if method is not None:
                event.connect(method)
    elif isinstance(connections, dict):
        # connections a dict, then match
        for event_name, method_name in connections.items():
            event = getattr(source.events, event_name)
            method = getattr(destination, method_name)
            event.connect(method)
    elif isinstance(connections, Iterable):
        # connections an iterable, then match to `_on_*_change`
        for event_name in connections:
            event = getattr(source.events, event_name)
            method = getattr(destination, f'_on_{event_name}_change')
            event.connect(method)
    else:
        raise ValueError('Connections not recognized')


def disconnect(source, destination, connections=None):
    if connections is None:
        # connection None then do automatically
        for em in source.events.emitters.values():
            for callback in em.callbacks:
                # Callback is a tuple of a weak reference and method name
                if (
                    isinstance(callback, tuple)
                    and callback[0] is destination.__weakref__
                ):
                    em.disconnect(callback)
    elif isinstance(connections, dict):
        # connections a dict, then match
        for event_name, method_name in connections.items():
            event = getattr(source.events, event_name)
            method = getattr(destination, method_name)
            event.disconnect(method)
    elif isinstance(connections, Iterable):
        # connections an iterable, then match to `_on_*_change`
        for event_name in connections:
            event = getattr(source.events, event_name)
            method = getattr(destination, f'_on_{event_name}_change')
            event.disconnect(method)
    else:
        raise ValueError('Connections not recognized')
