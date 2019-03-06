from napari.components.component import Component

signal = False

def test_listeners():

    component = Component()

    def listener(kwargs):
        global signal
        signal = kwargs['signal']
        print(kwargs)

    component.add_listener(listener)

    component._notify_listeners(signal=True)

    assert signal





