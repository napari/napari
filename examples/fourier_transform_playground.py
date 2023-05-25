"""
Fourier transform playground
============================

Generate an image by adding arbitrary 2D sine waves and observe
how the fourier transform's real and imaginary components are affected
by the changes. Threading is used to smoothly animate the waves.

.. tags:: interactivity, gui
"""


from time import sleep, time

import numpy as np
from magicgui import magic_factory
from scipy.fft import fft2, fftshift

import napari
from napari.qt.threading import thread_worker

IMAGE_SIZE = 100
FPS = 20

# meshgrid used to calculate the 2D sine waves
x = np.arange(IMAGE_SIZE) - IMAGE_SIZE / 2
X, Y = np.meshgrid(x, x)


# set up viewer with grid-mode enabled
viewer = napari.Viewer()
viewer.grid.enabled = True


def wave_2d(wavelength, angle, phase_shift, speed):
    """
    Generate a 2D sine wave based on angle and wavelength.

    The wave phase if offset by phase_shift and the current time,
    multiplied by an arbitrary speed value; this generates an animated
    wave if called repeatedly.
    """
    angle = np.deg2rad(angle)
    phase_shift = np.deg2rad(phase_shift)
    wave = 2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
    return np.sin(wave + phase_shift + (time() * speed))


def update_layer(name, data, **kwargs):
    """
    Update a layer in the viewer with new data.

    If data is None, then the layer is removed.
    If the layer is not present, it's added to the viewer.
    """
    if data is None:
        if name in viewer.layers:
            viewer.layers.pop(name)
    elif name not in viewer.layers:
        viewer.add_image(data, name=name, **kwargs)
    else:
        viewer.layers[name].data = data


def combine_and_set_data(waves):
    """
    Merge 2D waves, calculate the FT and update the viewer.
    """
    if not waves:
        # this happens on yielding from the thread, no need to update anything
        return

    to_add = [d for d in waves.values() if d is not None]

    if to_add:
        mean = np.mean(to_add, axis=0)
        ft = fftshift(fft2(mean))
        power_spectrum = abs(ft)
        phase = ft.imag
    else:
        mean = power_spectrum = phase = None

    update_layer('sum', mean)
    update_layer('power_spectrum', power_spectrum)
    update_layer('phase', phase, colormap=('red', 'black', 'blue'))

    for name, data in waves.items():
        update_layer(f'wave {name}', data)


@thread_worker(connect={"yielded": combine_and_set_data})
def update_viewer():
    # keep track of each wave in a dictionary by id, this way we can modify/remove
    # existing waves or add new ones
    waves = {}
    new_params = None
    while True:
        sleep(1 / FPS)
        # see https://napari.org/stable/guides/threading.html#full-two-way-communication
        # this receives new_params from thread.send() and yields waves for the `yielded` callback
        new_params = yield waves
        if new_params is not None:
            # note that these come from thread.send() in moving_wave()!
            wave_id, *args = new_params
            waves[wave_id] = args
        # remove (set value to None) any wave with wavelength 0, but generate the rest
        yield {
            wave_id: wave_2d(wavelength, angle, phase_shift, speed) if wavelength else None
            for wave_id, (wavelength, angle, phase_shift, speed) in waves.items()
        }


# start the thread responsible for updating the viewer
thread = update_viewer()


@magic_factory(
    auto_call=True,
    wavelength={'widget_type': 'Slider', 'min': 0, 'max': IMAGE_SIZE},
    angle={'widget_type': 'Slider', 'min': 0, 'max': 180},
    phase_shift={'widget_type': 'Slider', 'min': 0, 'max': 180},
    speed={'widget_type': 'FloatSlider', 'min': -10, 'max': 10, 'step': 0.1},
)
def moving_wave(
    wave_id: int = 0,
    wavelength: int = IMAGE_SIZE // 2,
    angle: int = 0,
    phase_shift: int = 0,
    speed: float = 1,
    run=True,
):
    """
    Send new parameters to the listening thread to update the 2D waves.

    The `run` checkbox can be disabled to stop sending values to the thread
    while changing parameters.
    """
    if run:
        thread.send((wave_id, wavelength, angle, phase_shift, speed))


# add the widget to the window
viewer.window.add_dock_widget(moving_wave())
