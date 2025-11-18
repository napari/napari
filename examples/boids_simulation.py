"""
Interactive Control Over Parallel Computations
==============================================

Demonstrates the ability to drive and manipulate computations/simulations
from within napari.  The basis of the compute-intensive simulation in this
example is a 3D implementation of Boids [1]_ which is accelerated by using
multiple CPU cores either on a single machine (e.g. a laptop) via the
multiprocessing [2]_ module from the Python Standard Library or across
multiple nodes (e.g. a Linux cluster or even a supercomputer) via
Dragon [3]_ which plugs into standard multiprocessing to extend it to
execute across multiple nodes.

Within are examples of how to start/pause/restart/stop computation from
the gui via QTWidgets and napari's thread_worker.

.. [1] http://www.red3d.com/cwr/boids/
.. [2] https://docs.python.org/3/library/multiprocessing.html
.. [3] https://github.com/DragonHPC/dragon

.. tags:: interactivity, gui
"""

try:
    import dragon
    import multiprocessing as mp  # Must be after import of dragon

    mp.set_start_method("dragon")
except (ImportError, ValueError):
    if __name__ == "__main__":
        print(f"Note: dragon unavailable; using standard multiprocessing")
except RuntimeError as re:
    print(f"Warning: {re}")
finally:
    import multiprocessing as mp

from functools import partial
import sys
import time
import numpy as np
import qtpy.QtWidgets
import napari


viewer, state = None, None


def radarray(N):
    return np.random.uniform(0, 2 * np.pi, N)


def count(mask, n):
    return np.maximum(mask.sum(axis=1), 1).reshape(n, 1)


def limit_acceleration(steer, n, maxacc=0.03):
    norm = np.sqrt((steer * steer).sum(axis=1)).reshape(n, 1)
    np.multiply(steer, maxacc / norm, out=steer, where=norm > maxacc)
    return norm, steer


class Boids3D:
    """Implementation of Boids in 3D.  Ref: http://www.red3d.com/cwr/boids/

    Directly inspired by implementation for HoloViz by jlstevens.
    Ref: https://examples.holoviz.org/gallery/boids/boids.html
    """

    def __init__(self, N=500, width=400, height=400, depth=400):
        self.width, self.height, self.depth, self.iteration = width, height, depth, 0
        self.vel = np.vstack(
            [
                np.cos(radarray(N)),
                np.sin(radarray(N)),
                np.cos(radarray(N)),
            ]
        ).T
        r = min(width, height) / 2 * np.random.uniform(0, 1, N)
        self.pos = np.vstack(
            [
                width / 2 + np.cos(radarray(N)) * r / 4,
                height / 2 + np.sin(radarray(N)) * r,
                depth / 2 + np.sin(radarray(N)) * r,
            ]
        ).T

    def flock(
        self,
        min_vel=0.5,
        max_vel=2.0,
        veryclose_radius=25,
        kindaclose_radius=50,
        slice_tot=1,
        pool=None,
    ):
        assert veryclose_radius <= kindaclose_radius
        kwargs = dict(
            pos=self.pos,
            vel=self.vel,
            width=self.width,
            height=self.height,
            depth=self.depth,
            min_vel=min_vel,
            max_vel=max_vel,
            veryclose_radius=veryclose_radius,
            kindaclose_radius=kindaclose_radius,
        )
        if pool is None:
            self.pos[:], self.vel[:] = self.compute_flock_step_forward(**kwargs)
        else:
            tasks = [
                pool.apply_async(
                    self.compute_flock_step_forward,
                    kwds=(kwargs | {"slice_idx": slice_idx, "slice_tot": slice_tot}),
                    # kwargs=(kwargs | {"slice_idx": slice_idx, "slice_tot": slice_tot}),
                )
                for slice_idx in range(slice_tot)
            ]
            for slice_idx in range(slice_tot):
                self.pos[slice_idx::slice_tot], self.vel[slice_idx::slice_tot] = tasks[
                    slice_idx
                ].get(timeout=30)
        self.iteration += 1

    @staticmethod
    def compute_flock_step_forward(
        pos,
        vel,
        width,
        height,
        depth,
        min_vel=0.5,
        max_vel=2.0,
        veryclose_radius=25,
        kindaclose_radius=50,
        slice_idx=0,
        slice_tot=1,
    ):
        assert slice_tot > slice_idx >= 0
        all_pos, all_vel = pos, vel
        pos = all_pos[slice_idx::slice_tot]
        vel = all_vel[slice_idx::slice_tot]
        n_pos, n_all_pos = len(pos), len(all_pos)

        dx = np.subtract.outer(pos[:, 0], all_pos[:, 0])
        dy = np.subtract.outer(pos[:, 1], all_pos[:, 1])
        dz = np.subtract.outer(pos[:, 2], all_pos[:, 2])

        dist = np.hypot(np.hypot(dx, dy), dz)
        mask_veryclose, mask_kindaclose = (
            (dist > 0) * (dist < veryclose_radius),
            (dist > 0) * (dist < kindaclose_radius),
        )
        target = np.dstack((dx, dy, dz))
        target = np.divide(
            target,
            dist.reshape(n_pos, n_all_pos, 1) ** 2,
            out=target,
            where=dist.reshape(n_pos, n_all_pos, 1) != 0,
        )
        steer = (target * mask_veryclose.reshape(n_pos, n_all_pos, 1)).sum(
            axis=1
        ) / count(mask_veryclose, n_pos)
        norm = np.sqrt((steer * steer).sum(axis=1)).reshape(n_pos, 1)
        steer = max_vel * np.divide(steer, norm, out=steer, where=norm != 0) - vel
        norm, separation = limit_acceleration(steer, n_pos)
        target = np.dot(mask_kindaclose, all_vel) / count(mask_kindaclose, n_pos)
        norm = np.sqrt((target * target).sum(axis=1)).reshape(n_pos, 1)
        target = max_vel * np.divide(target, norm, out=target, where=norm != 0)
        steer = target - vel
        norm, alignment = limit_acceleration(steer, n_pos)
        target = np.dot(mask_kindaclose, all_pos) / count(mask_kindaclose, n_pos)
        desired = target - pos
        norm = np.sqrt((desired * desired).sum(axis=1)).reshape(n_pos, 1)
        desired *= max_vel / norm
        steer = desired - vel
        norm, cohesion = limit_acceleration(steer, n_pos)
        vel += 1.5 * separation + alignment + cohesion
        norm = np.sqrt((vel * vel).sum(axis=1)).reshape(n_pos, 1)
        np.multiply(vel, max_vel / norm, out=vel, where=norm > max_vel)
        np.multiply(vel, min_vel / norm, out=vel, where=norm < min_vel)
        pos += vel + (width, height, depth)
        pos %= (width, height, depth)

        return pos, vel


@napari.qt.thread_worker
def iterate_flock_forever(boids):
    while True:
        boids.flock()
        yield boids.pos


@napari.qt.thread_worker
def iterate_accelerated_flock_forever(boids, num_workers=8):
    with mp.Pool(num_workers) as pool:
        while True:
            boids.flock(pool=pool, slice_tot=num_workers)
            yield boids.pos


timestamp_last_update = time.monotonic()
boids_layer = None


def update_points_in_display(boids):
    global timestamp_last_update
    if boids_layer:
        boids_layer.data = boids.pos
        timestamp_this_update = time.monotonic()
        elapsed_time_update, timestamp_last_update = (
            (timestamp_this_update - timestamp_last_update),
            timestamp_this_update,
        )
        boids_layer.name = f"Iter_{boids.iteration} ({elapsed_time_update:0.3f}s/iter) {state['num_boids']}-on-{state['num_workers']}-procs"


def add_relaunch_buttons():
    global line_edit_num_boids, line_edit_num_procs, checkbox_hide_prev
    remove_execution_control_buttons()

    info_label_num_boids = qtpy.QtWidgets.QLabel()
    info_label_num_boids.setText('Number of "Boids"')
    line_edit_num_boids = qtpy.QtWidgets.QLineEdit()
    line_edit_num_boids.setText(str(state.get("num_boids", 1500)))
    line_edit_num_boids.setValidator(qtpy.QtGui.QDoubleValidator())
    info_label_num_procs = qtpy.QtWidgets.QLabel()
    info_label_num_procs.setText("Number of Processes")
    line_edit_num_procs = qtpy.QtWidgets.QLineEdit()
    line_edit_num_procs.setText(str(state.get("num_workers", 1)))
    line_edit_num_procs.setValidator(qtpy.QtGui.QDoubleValidator())
    checkbox_hide_prev = qtpy.QtWidgets.QCheckBox()
    checkbox_hide_prev.setText("Hide Previous Sim")
    checkbox_hide_prev.setChecked(True)
    launch_button = qtpy.QtWidgets.QPushButton("Launch New Sim!")
    launch_button.clicked.connect(update_state_then_launch_main)

    widget = qtpy.QtWidgets.QWidget()
    layout = qtpy.QtWidgets.QVBoxLayout()
    widget.setLayout(layout)
    layout.addWidget(info_label_num_boids)
    layout.addWidget(line_edit_num_boids)
    layout.addWidget(info_label_num_procs)
    layout.addWidget(line_edit_num_procs)
    layout.addWidget(checkbox_hide_prev)
    layout.addWidget(launch_button)
    viewer.window.add_dock_widget(widget)


def get_values_from_relaunch_buttons():
    global line_edit_num_boids, line_edit_num_procs, checkbox_hide_prev
    return (
        int(line_edit_num_boids.text()),
        int(line_edit_num_procs.text()),
        checkbox_hide_prev.checkState(),
    )


def action_start_button(worker):
    global boids_layer
    for layer in viewer.layers:
        if layer.name.startswith("Boids3D"):
            boids_layer = layer
            break
    else:
        boids_layer = layer  # Hope for the best.
    worker.start()


def action_terminate_button(worker):
    global boids_layer
    worker.quit()  # Beware, update_points_in_display() may still be running.
    boids_layer = None


def add_execution_control_buttons(viewer, worker):
    start_button = qtpy.QtWidgets.QPushButton("Start Sim")
    start_button.clicked.connect(partial(action_start_button, worker))
    pause_button = qtpy.QtWidgets.QPushButton("Pause")
    pause_button.clicked.connect(worker.pause)
    resume_button = qtpy.QtWidgets.QPushButton("Resume")
    resume_button.clicked.connect(worker.resume)
    stop_button = qtpy.QtWidgets.QPushButton("Terminate")
    stop_button.clicked.connect(partial(action_terminate_button, worker))
    all_buttons = (start_button, pause_button, resume_button, stop_button)
    for button in all_buttons:
        worker.finished.connect(button.clicked.disconnect)
        viewer.window.add_dock_widget(button)
    return all_buttons


def remove_execution_control_buttons():
    viewer.window.remove_dock_widget("all")


def update_state_then_launch_main():
    global state
    num_boids, num_procs, hide_prior = get_values_from_relaunch_buttons()
    if hide_prior:
        viewer.layers[-1].visible = False
    state["num_boids"] = num_boids
    state["num_workers"] = num_procs
    state["parallel"] = True if num_procs > 1 else False
    return main(**state)


def main(
    num_boids=500, width=600, height=600, depth=800, parallel=False, num_workers=1
):
    global viewer, state
    state = dict(
        num_boids=num_boids,
        width=width,
        height=height,
        depth=depth,
        parallel=parallel,
        num_workers=num_workers,
    )
    boids = Boids3D(num_boids, width, height, depth)

    title = f"napari+{'Dragon' if 'dragon' in sys.modules else 'multiprocessing'}"
    viewer = napari.current_viewer() or napari.Viewer(title=title, ndisplay=3)
    viewer.dims.ndisplay = 3  # In case we're reusing an existing viewer.
    remove_execution_control_buttons()
    viewer.add_points(
        boids.pos, 3, name="Boids3D"
    )  # Show initial starting points/data.

    if not parallel:
        worker = iterate_flock_forever(boids)
    else:
        worker = iterate_accelerated_flock_forever(boids, num_workers=num_workers)
    worker.yielded.connect(partial(update_points_in_display, boids))

    add_execution_control_buttons(viewer, worker)
    worker.finished.connect(add_relaunch_buttons)

    return boids, viewer, worker


if __name__ == "__main__":
    b, v, w = main(1500, parallel=True, num_workers=4)
    napari.run()
