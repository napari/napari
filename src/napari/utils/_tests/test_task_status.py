from unittest.mock import Mock

from napari.utils.task_status import (
    Status,
    register_task_status,
    task_status_manager,
    update_task_status,
)


def test_task_status():
    """test task status registration and update using the utils.task_status module."""
    # Check task status registration
    cancel_callback_mock = Mock()
    task_status_id = register_task_status(
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
    update_task_status(
        task_status_id, Status.DONE, description='Register task status done'
    )
    assert not task_status_manager.is_busy()
    assert task_status_manager.get_status() == []
    update_task_status(
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
