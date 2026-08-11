"""Hardware detection and device selection utility for local CPU vs cloud GPU execution."""

from typing import Dict, Any
import tensorflow as tf
from medvision.utils.logger import get_logger

logger = get_logger("medvision.device")


def get_execution_device(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Detect available hardware (CPU vs GPU) and return device execution details.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Dict containing device type, GPU count, device name, and execution strategy details.
    """
    force_device = "AUTO"
    if config:
        force_device = config.get("device", {}).get("force_device", "AUTO")

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0

    if force_device == "CPU":
        selected_device = "CPU"
        device_name = "/CPU:0"
    elif force_device == "GPU" and gpu_available:
        selected_device = "GPU"
        device_name = gpus[0].name
    else:  # AUTO mode
        if gpu_available:
            selected_device = "GPU"
            device_name = gpus[0].name
        else:
            selected_device = "CPU"
            device_name = "/CPU:0"

    logger.info(
        f"Hardware Detection Result: Selected [{selected_device}] device ({device_name}). "
        f"GPUs Available: {len(gpus)}"
    )

    return {
        "device_type": selected_device,
        "device_name": device_name,
        "gpu_count": len(gpus),
        "is_gpu": selected_device == "GPU",
    }
