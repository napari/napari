def connect(source, destination, connections):
    for event_name in connections:
        event = getattr(source.events, event_name)
        method = getattr(destination, f'_on_{event_name}_change')
        event.connect(method)


def disconnect(source, destination, connections):
    for event_name in connections:
        event = getattr(source.events, event_name)
        method = getattr(destination, f'_on_{event_name}_change')
        event.disconnect(method)
