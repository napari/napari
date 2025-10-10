"""Camera inertia for smooth panning and rotation animation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer

if TYPE_CHECKING:
    import numpy.typing as npt

    from napari.components import ViewerModel


class InertiaState(Enum):
    """State of the inertia system."""

    IDLE = auto()  # Not tracking or animating
    TRACKING = auto()  # Tracking mouse movement during drag
    ANIMATING = auto()  # Playing animation after release


@dataclass
class InertiaConfig:
    """Configuration for camera inertia behavior.

    Parameters
    ----------
    pan_friction : float
        Pan decay rate per second. Higher values cause faster deceleration.
        Default: 5.0
    pan_damping : float
        Fraction of velocity to apply for panning motion (0-1).
        Lower values reduce initial velocity. Default: 0.6
    pan_max_speed : float
        Maximum pan velocity in world units/second. Default: 200.0
    pan_min_speed : float
        Minimum pan speed to trigger inertia animation. Default: 4.0
    pan_stop_speed : float
        Pan speed threshold below which animation stops. Default: 2.5
    rotate_friction : float
        Rotation decay rate per second. Higher values cause faster deceleration.
        Default: 7.0 (slightly faster than pan for less dizziness)
    rotate_damping : float
        Fraction of velocity to apply for rotation (0-1).
        Rotation is usually more sensitive than panning. Default: 0.4
    rotate_max_speed : float
        Maximum rotation velocity in degrees per second. Default: 120.0
    rotate_min_speed : float
        Minimum rotation speed to trigger rotation inertia. Default: 1.5
    rotate_stop_speed : float
        Rotation speed threshold below which animation stops. Default: 1.0
    max_dt : float
        Maximum time (seconds) between last movement and release
        to trigger inertia. Prevents stale velocity from being used.
        Default: 0.1
    timer_interval_ms : int
        Interval in milliseconds for animation timer (~60 FPS). Default: 16
    """

    pan_friction: float = 5.0
    pan_damping: float = 0.6
    pan_max_speed: float = 200.0
    pan_min_speed: float = 4.0
    pan_stop_speed: float = 2.5
    rotate_friction: float = 7.0
    rotate_damping: float = 0.4
    rotate_max_speed: float = 120.0
    rotate_min_speed: float = 1.5
    rotate_stop_speed: float = 1.0
    max_dt: float = 0.1
    timer_interval_ms: int = 16

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pan_friction < 0:
            raise ValueError('pan_friction must be non-negative')
        if self.rotate_friction < 0:
            raise ValueError('rotate_friction must be non-negative')
        if not 0 <= self.pan_damping <= 1:
            raise ValueError('pan_damping must be between 0 and 1')
        if not 0 <= self.rotate_damping <= 1:
            raise ValueError('rotate_damping must be between 0 and 1')
        if self.pan_max_speed <= 0:
            raise ValueError('pan_max_speed must be positive')
        if self.rotate_max_speed <= 0:
            raise ValueError('rotate_max_speed must be positive')
        if self.pan_min_speed < 0:
            raise ValueError('pan_min_speed must be non-negative')
        if self.rotate_min_speed < 0:
            raise ValueError('rotate_min_speed must be non-negative')
        if self.pan_stop_speed < 0:
            raise ValueError('pan_stop_speed must be non-negative')
        if self.rotate_stop_speed < 0:
            raise ValueError('rotate_stop_speed must be non-negative')
        if self.max_dt <= 0:
            raise ValueError('max_dt must be positive')
        if self.timer_interval_ms <= 0:
            raise ValueError('timer_interval_ms must be positive')


class CameraInertia:
    """Manages camera inertia for smooth panning and rotation animation.

    This class tracks mouse movement during drag operations and applies
    a smooth deceleration animation when the mouse is released, creating
    an inertial "coast" effect similar to mobile touch interfaces.

    Parameters
    ----------
    viewer : ViewerModel
        The napari viewer instance.
    config : InertiaConfig, optional
        Configuration for inertia behavior. If None, uses defaults.

    Attributes
    ----------
    enabled : bool
        Whether inertia is enabled. Can be toggled at runtime.
    """

    def __init__(
        self, viewer: ViewerModel, config: InertiaConfig | None = None
    ) -> None:
        self._viewer = viewer
        self._config = config or InertiaConfig()
        self._state = InertiaState.IDLE

        # Tracking state
        self._last_pos: npt.NDArray[np.floating] | None = None
        self._last_angles: npt.NDArray[np.floating] | None = None
        self._last_time: float | None = None

        # Animation state
        self._pan_velocity: npt.NDArray[np.floating] | None = None
        self._rotate_velocity: npt.NDArray[np.floating] | None = None

        # Animation timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._animation_step)

        # Public control
        self.enabled = True

    @property
    def state(self) -> InertiaState:
        """Current state of the inertia system."""
        return self._state

    @property
    def is_animating(self) -> bool:
        """Whether inertia animation is currently playing."""
        return self._state == InertiaState.ANIMATING

    def on_press(self) -> None:
        """Handle mouse press event.

        Stops any ongoing animation and resets tracking state.
        """
        if not self.enabled:
            return

        self.stop()
        self._state = InertiaState.IDLE

    def on_drag(self) -> None:
        """Handle mouse drag event.

        Tracks position and angles for velocity calculation.
        Should be called during mouse movement while dragging.
        """
        if not self.enabled:
            return

        pos = np.array(self._viewer.camera.center, dtype=np.float64)
        angles = np.array(self._viewer.camera.angles, dtype=np.float64)
        now = perf_counter()

        self._last_pos = pos
        self._last_angles = angles
        self._last_time = now
        self._state = InertiaState.TRACKING

    def on_release(self) -> None:
        """Handle mouse release event.

        Calculates velocity from tracked movement and starts inertia
        animation if velocity is significant enough.
        """
        if not self.enabled or self._state != InertiaState.TRACKING:
            self._reset_tracking()
            return

        if self._last_pos is None or self._last_time is None:
            self._reset_tracking()
            return

        current_pos = np.array(self._viewer.camera.center, dtype=np.float64)
        current_angles = np.array(self._viewer.camera.angles, dtype=np.float64)
        current_time = perf_counter()
        dt = current_time - self._last_time

        # Only start inertia if the release is recent enough
        if not (0.001 < dt < self._config.max_dt):
            self._reset_tracking()
            return

        # Calculate and apply pan velocity
        pan_velocity = self._calculate_pan_velocity(current_pos, dt)

        # Calculate and apply rotation velocity (3D only)
        rotate_velocity = self._calculate_rotate_velocity(current_angles, dt)

        # Start animation if either velocity is significant
        if pan_velocity is not None or rotate_velocity is not None:
            self._pan_velocity = pan_velocity
            self._rotate_velocity = rotate_velocity
            self._last_time = current_time
            self._state = InertiaState.ANIMATING
            self._timer.start(self._config.timer_interval_ms)
        else:
            self._reset_tracking()

    def stop(self) -> None:
        """Stop inertia animation and reset state."""
        if self._timer.isActive():
            self._timer.stop()
        self._pan_velocity = None
        self._rotate_velocity = None
        self._reset_tracking()
        if self._state == InertiaState.ANIMATING:
            self._state = InertiaState.IDLE

    def _reset_tracking(self) -> None:
        """Reset tracking state variables."""
        self._last_pos = None
        self._last_angles = None
        self._last_time = None
        if self._state == InertiaState.TRACKING:
            self._state = InertiaState.IDLE

    def _calculate_pan_velocity(
        self, current_pos: npt.NDArray[np.floating], dt: float
    ) -> npt.NDArray[np.floating] | None:
        """Calculate pan velocity for camera panning.

        Parameters
        ----------
        current_pos : np.ndarray
            Current camera center position.
        dt : float
            Time delta since last tracked position.

        Returns
        -------
        np.ndarray or None
            Pan velocity vector, or None if below threshold.
        """
        if self._last_pos is None:
            return None

        velocity = (current_pos - self._last_pos) / dt

        # Apply damping
        velocity = velocity * self._config.pan_damping

        # Cap maximum speed
        speed = np.linalg.norm(velocity)
        if speed > self._config.pan_max_speed:
            velocity = velocity * (self._config.pan_max_speed / speed)
            speed = self._config.pan_max_speed

        # Check if velocity is significant enough
        if speed < self._config.pan_min_speed:
            return None

        return velocity

    def _calculate_rotate_velocity(
        self, current_angles: npt.NDArray[np.floating], dt: float
    ) -> npt.NDArray[np.floating] | None:
        """Calculate rotation velocity for camera rotation.

        Parameters
        ----------
        current_angles : np.ndarray
            Current camera angles.
        dt : float
            Time delta since last tracked angles.

        Returns
        -------
        np.ndarray or None
            Rotation velocity vector (degrees/sec), or None if below threshold.
        """
        if self._last_angles is None:
            return None

        rotate_velocity = (current_angles - self._last_angles) / dt

        # Apply rotation-specific damping
        rotate_velocity = rotate_velocity * self._config.rotate_damping

        # Cap maximum rotation speed
        rotate_speed = np.linalg.norm(rotate_velocity)
        if rotate_speed > self._config.rotate_max_speed:
            rotate_velocity = rotate_velocity * (
                self._config.rotate_max_speed / rotate_speed
            )
            rotate_speed = self._config.rotate_max_speed

        # Check if rotation velocity is significant enough
        if rotate_speed < self._config.rotate_min_speed:
            return None

        return rotate_velocity

    def _animation_step(self) -> None:
        """Execute one step of inertia animation with friction."""
        # Verify we still have active velocities
        if self._pan_velocity is None and self._rotate_velocity is None:
            self.stop()
            return

        now = perf_counter()
        dt = now - self._last_time if self._last_time is not None else 0.0
        self._last_time = now

        # Apply pan velocity with pan-specific friction
        if self._pan_velocity is not None:
            pan_decay = np.exp(-self._config.pan_friction * dt)
            self._pan_velocity = self._pan_velocity * pan_decay
            displacement = self._pan_velocity * dt
            center = np.array(self._viewer.camera.center, dtype=np.float64)
            center = center + displacement
            self._viewer.camera.center = tuple(center)

            # Stop pan velocity if too small
            pan_speed = np.linalg.norm(self._pan_velocity)
            if pan_speed < self._config.pan_stop_speed:
                self._pan_velocity = None

        # Apply rotation velocity with rotation-specific friction (3D only)
        if self._rotate_velocity is not None:
            rotate_decay = np.exp(-self._config.rotate_friction * dt)
            self._rotate_velocity = self._rotate_velocity * rotate_decay
            angular_displacement = self._rotate_velocity * dt
            angles = np.array(self._viewer.camera.angles, dtype=np.float64)
            angles = angles + angular_displacement
            self._viewer.camera.angles = tuple(angles)

            # Stop rotation velocity if too small
            rotate_speed = np.linalg.norm(self._rotate_velocity)
            if rotate_speed < self._config.rotate_stop_speed:
                self._rotate_velocity = None

        # Stop timer if both velocities are done
        if self._pan_velocity is None and self._rotate_velocity is None:
            self.stop()
