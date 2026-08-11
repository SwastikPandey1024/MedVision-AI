"""Unit tests for hardware auto-detection utility."""

from medvision.utils.device import get_execution_device


def test_get_execution_device():
    """Verify device detection returns valid dictionary with device_type and status."""
    device_info = get_execution_device()
    assert "device_type" in device_info
    assert "device_name" in device_info
    assert "gpu_count" in device_info
    assert device_info["device_type"] in ("CPU", "GPU")
