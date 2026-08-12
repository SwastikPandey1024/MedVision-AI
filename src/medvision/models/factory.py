"""Extensible model factory for architecture selection (Custom CNN, DenseNet121, EfficientNetB0)."""

from typing import Tuple, Dict, Any, Optional
import keras
import tensorflow as tf

from medvision.models.baseline_cnn import build_custom_cnn
from medvision.models.densenet import build_densenet121
from medvision.utils.metrics import get_model_metrics
from medvision.utils.logger import get_logger

logger = get_logger("medvision.models.factory")


def get_distribution_strategy(force_single_gpu: bool = False) -> Tuple[tf.distribute.Strategy, int]:
    """Dynamically setup Multi-GPU MirroredStrategy or default single-device strategy.

    Args:
        force_single_gpu: If True, bypass MirroredStrategy.

    Returns:
        Tuple of (tf.distribute.Strategy, gpu_count).
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus)

    if gpu_count > 1 and not force_single_gpu:
        logger.info(f"Multi-GPU detected ({gpu_count} GPUs). Activating tf.distribute.MirroredStrategy().")
        strategy = tf.distribute.MirroredStrategy()
    elif gpu_count == 1:
        logger.info(f"Single GPU detected ({gpus[0].name}). Using default strategy.")
        strategy = tf.distribute.get_strategy()
    else:
        logger.info("No GPU detected. Running on CPU with default strategy.")
        strategy = tf.distribute.get_strategy()

    return strategy, gpu_count


def configure_mixed_precision(enable: bool = True) -> str:
    """Configures Keras mixed precision policy.

    Args:
        enable: If True, attempts to enable mixed_float16.

    Returns:
        Active mixed precision policy string.
    """
    if enable:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                keras.mixed_precision.set_global_policy("mixed_float16")
                logger.info("Mixed precision policy set to 'mixed_float16'.")
                return "mixed_float16"
            except Exception as e:
                logger.warning(f"Failed to enable mixed_float16: {e}. Falling back to float32.")
                keras.mixed_precision.set_global_policy("float32")
                return "float32"
        else:
            logger.info("No GPU available for mixed precision. Staying on 'float32'.")
            keras.mixed_precision.set_global_policy("float32")
            return "float32"
    else:
        keras.mixed_precision.set_global_policy("float32")
        return "float32"


def build_model(
    architecture: str = "densenet121",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    learning_rate: float = 1e-4,
    compile_model: bool = True,
    mixed_precision: bool = False,
    config: Optional[Dict[str, Any]] = None,
    strategy: Optional[tf.distribute.Strategy] = None,
) -> keras.Model:
    """Build and compile classification model based on selected architecture.

    Args:
        architecture: Model architecture ('custom_cnn', 'densenet121', 'efficientnetb0').
        input_shape: Input image dimensions.
        num_classes: Output class count (1 for binary).
        learning_rate: Initial learning rate for Adam optimizer.
        compile_model: If True, compiles model with loss, optimizer, and metrics.
        mixed_precision: If True, configures mixed_float16 policy.
        config: Master configuration dictionary.
        strategy: Optional pre-existing tf.distribute.Strategy instance.

    Returns:
        Keras Model instance (compiled if compile_model=True).
    """
    valid_archs = ("custom_cnn", "densenet121", "efficientnetb0")
    if architecture not in valid_archs:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {valid_archs}")

    # Configure mixed precision policy
    configure_mixed_precision(enable=mixed_precision)

    # Use supplied strategy instance if provided; do NOT re-instantiate MirroredStrategy
    if strategy is not None:
        active_strategy = strategy
        logger.info(f"Using supplied distribution strategy: {active_strategy.__class__.__name__} (id={id(active_strategy)})")
    else:
        active_strategy, _ = get_distribution_strategy()
        logger.info(f"Fallback strategy detected: {active_strategy.__class__.__name__} (id={id(active_strategy)})")

    with active_strategy.scope():
        if architecture == "custom_cnn":
            init_filters = 32
            dropout_rate = 0.3
            if config and "model" in config and "baseline" in config["model"]:
                init_filters = config["model"]["baseline"].get("initial_filters", 32)
                dropout_rate = config["model"]["baseline"].get("dropout_rate", 0.3)

            model = build_custom_cnn(
                input_shape=input_shape,
                initial_filters=init_filters,
                dropout_rate=dropout_rate,
                num_classes=num_classes,
            )

        elif architecture == "densenet121":
            dense_units = 256
            dropout_rate = 0.4
            weights = "imagenet"
            if config and "model" in config and "densenet121" in config["model"]:
                dense_units = config["model"]["densenet121"].get("dense_units", 256)
                dropout_rate = config["model"]["densenet121"].get("dropout_rate", 0.4)
                weights = config["model"]["densenet121"].get("weights", "imagenet")

            model = build_densenet121(
                input_shape=input_shape,
                weights=weights,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                freeze_backbone=True,
                num_classes=num_classes,
            )

        elif architecture == "efficientnetb0":
            inputs = keras.layers.Input(shape=input_shape)
            try:
                base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
            except Exception as e:
                logger.warning(f"Failed to load ImageNet weights for EfficientNetB0 ({e}). Instantiating with random weights.")
                base = keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=inputs, pooling="avg")
            base.trainable = False
            outputs = keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32")(base.output)
            model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0Extensible")

        if compile_model:
            lr_val = float(learning_rate) if isinstance(learning_rate, (int, float, str)) else learning_rate
            optimizer = keras.optimizers.Adam(learning_rate=lr_val, clipnorm=1.0)
            loss_fn = keras.losses.BinaryCrossentropy()
            metrics = get_model_metrics()

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
            )
            logger.info(f"Successfully compiled '{architecture}' model (LR={learning_rate}, clipnorm=1.0).")

    return model
