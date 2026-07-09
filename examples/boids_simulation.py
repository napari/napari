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
    import dragon  # isort:skip  # noqa: F401
    import multiprocessing as mp  # Must be after import of dragon

    mp.set_start_method("dragon")
except (ImportError, ValueError):
    if __name__ == "__main__":
        print("Note: dragon unavailable; using standard multiprocessing")
except RuntimeError as re:
    print(f"Warning: {re}")
finally:
    import multiprocessing as mp

import sys
import time

import numpy as np
import qtpy.QtGui
import qtpy.QtWidgets

import napari


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
        kwargs = {
            'pos': self.pos,
            'vel': self.vel,
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'min_vel': min_vel,
            'max_vel': max_vel,
            'veryclose_radius': veryclose_radius,
            'kindaclose_radius': kindaclose_radius,
        }
        if pool is None:
            self.pos[:], self.vel[:] = self.compute_flock_step_forward(**kwargs)
        else:
            tasks = [
                pool.apply_async(
                    self.compute_flock_step_forward,
                    kwds=(kwargs | {"slice_idx": slice_idx, "slice_tot": slice_tot}),
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
control_widget = None


def update_points_in_display(boids, state):
    global timestamp_last_update
    if boids_layer is not None:
        boids_layer.data = boids.pos
        timestamp_this_update = time.monotonic()
        elapsed_time_update, timestamp_last_update = (
            (timestamp_this_update - timestamp_last_update),
            timestamp_this_update,
        )
        boids_layer.name = (
            f"Iter_{boids.iteration} ({elapsed_time_update:.3f}s/iter) "
            f"{state['num_boids']}-boids-on-{state['num_workers']}-procs"
        )


class BoidsControlWidget(qtpy.QtWidgets.QWidget):
    """Dock widget with controls for a running Boids simulation.

    All controls are built once. Button enabled states and text change as
    the simulation moves between idle, running, and finished states.
    """

    def __init__(self, viewer, worker, state):
        super().__init__()
        self._viewer = viewer
        self._worker = worker
        self._state = state
        self._paused = False

        layout = qtpy.QtWidgets.QVBoxLayout(self)

        form = qtpy.QtWidgets.QFormLayout()
        self._edit_num_boids = qtpy.QtWidgets.QLineEdit(
            str(state.get("num_boids", 1500))
        )
        self._edit_num_boids.setValidator(qtpy.QtGui.QIntValidator(1, 100_000))
        self._edit_num_procs = qtpy.QtWidgets.QLineEdit(
            str(state.get("num_workers", 1))
        )
        self._edit_num_procs.setValidator(qtpy.QtGui.QIntValidator(1, 256))
        form.addRow("Number of Boids", self._edit_num_boids)
        form.addRow("Number of Processes", self._edit_num_procs)
        layout.addLayout(form)

        self._hide_prev_check = qtpy.QtWidgets.QCheckBox("Hide previous sim")
        self._hide_prev_check.setChecked(True)
        self._hide_prev_check.setVisible(False)
        layout.addWidget(self._hide_prev_check)

        self._start_btn = qtpy.QtWidgets.QPushButton("Start Sim")
        self._pause_btn = qtpy.QtWidgets.QPushButton("Pause")
        self._stop_btn = qtpy.QtWidgets.QPushButton("Terminate")
        for btn in (self._start_btn, self._pause_btn, self._stop_btn):
            layout.addWidget(btn)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)

        self._start_btn.clicked.connect(self._on_start)
        self._pause_btn.clicked.connect(self._on_pause_resume)
        self._stop_btn.clicked.connect(self._on_stop)
        worker.finished.connect(self._on_worker_finished)

    def _on_start(self):
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._edit_num_boids.setEnabled(False)
        self._edit_num_procs.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)

    def _on_pause_resume(self):
        if self._paused:
            self._worker.resume()
            self._pause_btn.setText("Pause")
        else:
            self._worker.pause()
            self._pause_btn.setText("Resume")
        self._paused = not self._paused

    def _on_stop(self):
        global boids_layer
        self._worker.quit()
        boids_layer = None

    def _on_worker_finished(self):
        self._edit_num_boids.setEnabled(True)
        self._edit_num_procs.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")
        self._stop_btn.setEnabled(False)
        self._hide_prev_check.setVisible(True)
        self._start_btn.setText("Launch New Sim!")
        self._start_btn.setEnabled(True)
        self._start_btn.clicked.disconnect(self._on_start)
        self._start_btn.clicked.connect(self._on_relaunch)

    def _on_relaunch(self):
        num_boids = int(self._edit_num_boids.text())
        num_procs = int(self._edit_num_procs.text())
        if self._hide_prev_check.isChecked() and self._viewer.layers:
            self._viewer.layers[-1].visible = False
        self._state.update(
            num_boids=num_boids,
            num_workers=num_procs,
            parallel=num_procs > 1,
        )
        main(**self._state)


def main(
    num_boids=500, width=600, height=600, depth=800, parallel=False, num_workers=1
):
    global boids_layer, control_widget
    state = {
        'num_boids': num_boids,
        'width': width,
        'height': height,
        'depth': depth,
        'parallel': parallel,
        'num_workers': num_workers,
    }
    boids = Boids3D(num_boids, width, height, depth)

    title = f"napari+{'Dragon' if 'dragon' in sys.modules else 'multiprocessing'}"
    viewer = napari.current_viewer() or napari.Viewer(title=title, ndisplay=3)
    viewer.dims.ndisplay = 3  # ensure 3-D display when reusing an existing viewer

    if control_widget is not None:
        viewer.window.remove_dock_widget(control_widget)
        control_widget = None

    boids_layer = viewer.add_points(
        boids.pos, 3, name="Boids3D"
    )

    if not parallel:
        worker = iterate_flock_forever(boids)
    else:
        worker = iterate_accelerated_flock_forever(boids, num_workers=num_workers)
    worker.yielded.connect(lambda _: update_points_in_display(boids, state))

    control_widget = BoidsControlWidget(viewer, worker, state)
    viewer.window.add_dock_widget(control_widget, area='right', name='Boids Controls')

    return boids, viewer, worker


if __name__ == "__main__":
    b, v, w = main(1500, parallel=True, num_workers=4)
    napari.run()
