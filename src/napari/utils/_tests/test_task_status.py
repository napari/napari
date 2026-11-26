from unittest.mock import Mock

from napari.utils.task_status import Status, TaskStatusManager


def test_task_status():
    """test task status registration and update using the utils.task_status module."""
    task_status_manager = TaskStatusManager()

    # Check task status registration
    cancel_callback_mock = Mock()
    task_status_id = task_status_manager.register_task_status(
        'test-task-status',
        Status.BUSY,
        'Register task status busy',
        cancel_callback=cancel_callback_mock,
    )
    assert task_status_manager.is_busy()
    assert task_status_manager.get_status() == [
        'test-task-status: Register task status busy'
    ]

    # Check task status update
    task_status_manager.update_task_status(
        task_status_id,
        Status.COMPLETED,
        description='Register task status completed',
    )
    assert not task_status_manager.is_busy()
    assert task_status_manager.get_status() == []
    task_status_manager.update_task_status(
        task_status_id,
        Status.PENDING,
        description='Register task status pending',
    )
    assert task_status_manager.is_busy()
    assert task_status_manager.get_status() == [
        'test-task-status: Register task status pending'
    ]

    # Check cancel behavior
    task_status_manager.cancel_all()
    cancel_callback_mock.assert_called_once()
    assert task_status_manager.get_status() == []
    assert not task_status_manager.is_busy()
