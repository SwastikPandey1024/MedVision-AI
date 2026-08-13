"""Model architecture visualization utility for MedVision-AI.

Generates model summary TXT files and architecture block diagrams (SVG/PNG)
for local inspection without requiring a GPU.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import os
import datetime
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras
import tensorflow as tf

from medvision.config.settings import get_project_root
from medvision.models.factory import build_model
from medvision.utils.logger import get_logger

logger = get_logger("medvision.utils.visualization")


def generate_model_summary_txt(model: keras.Model, output_path: Path) -> str:
    """Generate detailed model summary text report and save to TXT file.

    Args:
        model: Keras Model instance.
        output_path: Target text file path.

    Returns:
        Formatted summary text string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append(f"MEDVISION-AI MODEL ARCHITECTURE REPORT: {model.name}")
    lines.append(f"Generated At: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("--- LAYER SUMMARY TABLE ---")
    header = f"{'Idx':<5} | {'Layer Name':<30} | {'Layer Type':<20} | {'Output Shape':<22} | {'Param #':<10} | {'Trainable':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    total_params = 0
    trainable_params = 0

    for idx, layer in enumerate(model.layers):
        l_name = layer.name[:30]
        l_type = layer.__class__.__name__[:20]
        try:
            out_shape = str(layer.output_shape)
        except Exception:
            out_shape = "Dynamic"
        
        p_count = layer.count_params()
        t_status = "YES" if layer.trainable else "NO"
        
        total_params += p_count
        if layer.trainable:
            trainable_params += p_count

        lines.append(f"{idx:<5} | {l_name:<30} | {l_type:<20} | {out_shape:<22} | {p_count:<10,} | {t_status:<10}")

        # Unpack nested backbone layers if Functional / Sequential base model
        if hasattr(layer, "layers") and isinstance(layer, (keras.Model, keras.src.models.Functional)):
            lines.append(f"      └─ Submodel '{layer.name}' with {len(layer.layers)} internal layers:")
            for sub_idx, sub_layer in enumerate(layer.layers[:15]):  # Show first 15 sublayers snippet
                sub_p = sub_layer.count_params()
                sub_t = "YES" if sub_layer.trainable else "NO"
                lines.append(
                    f"         [{sub_idx:03d}] {sub_layer.name[:25]:<25} | {sub_layer.__class__.__name__:<18} | "
                    f"Params: {sub_p:<8,} | Trainable: {sub_t}"
                )
            if len(layer.layers) > 15:
                lines.append(f"         ... [{len(layer.layers) - 15} more sublayers inside {layer.name}] ...")

    non_trainable_params = total_params - trainable_params
    estimated_size_mb = (total_params * 4) / (1024 * 1024)

    lines.append("")
    lines.append("=" * 80)
    lines.append("--- PARAMETER COUNT SUMMARY ---")
    lines.append(f"Model Identifier      : {model.name}")
    lines.append(f"Input Shape           : {model.input_shape}")
    lines.append(f"Output Shape          : {model.output_shape}")
    lines.append(f"Total Parameters      : {total_params:,}")
    lines.append(f"Trainable Parameters  : {trainable_params:,}")
    lines.append(f"Non-trainable Params  : {non_trainable_params:,}")
    lines.append(f"Estimated Weights Size: {estimated_size_mb:.2f} MB (FP32 precision)")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(f"Saved model summary text to: {output_path}")
    return summary_text


def _render_matplotlib_diagram(model: keras.Model, svg_path: Path, png_path: Path) -> None:
    """Fallback matplotlib architecture block diagram generator.

    Renders a clean SVG and PNG layer diagram without external C++ Graphviz dependencies.
    """
    layers_info = []

    for layer in model.layers:
        l_type = layer.__class__.__name__
        l_name = layer.name
        try:
            out_shape = str(layer.output_shape)
        except Exception:
            out_shape = "(Dynamic)"
        
        p_count = layer.count_params()
        layers_info.append({
            "name": l_name,
            "type": l_type,
            "shape": out_shape,
            "params": p_count,
            "trainable": layer.trainable,
        })

    num_blocks = len(layers_info)
    max_blocks_for_render = min(num_blocks, 40)
    render_width = max(10, max_blocks_for_render * 2.2)
    fig, ax = plt.subplots(figsize=(render_width, 6))
    ax.axis("off")

    color_map = {
        "InputLayer": "#4A90E2",
        "Conv2D": "#50E3C2",
        "BatchNormalization": "#B8E986",
        "ReLU": "#F5A623",
        "Activation": "#F5A623",
        "MaxPooling2D": "#7ED321",
        "GlobalAveragePooling2D": "#9013FE",
        "Dropout": "#D0021B",
        "Dense": "#BD10E0",
        "Functional": "#4A90E2",
        "Model": "#4A90E2",
    }

    x_offset = 0.5
    box_width = 1.8
    box_height = 3.5

    ax.set_xlim(0, max(12, render_width + 1))
    ax.set_ylim(0, 7)

    ax.text(
        0.5, 6.4, f"Architecture Diagram: {model.name}",
        fontsize=14, fontweight="bold", ha="left", va="center"
    )
    ax.text(
        0.5, 6.0, f"Total Params: {model.count_params():,} | Input: {model.input_shape} | Output: {model.output_shape}",
        fontsize=10, color="#555555", ha="left", va="center"
    )

    if num_blocks > max_blocks_for_render:
        ax.text(
            0.5, 0.6,
            f"Diagram truncated to first {max_blocks_for_render} of {num_blocks} layers for readability.",
            fontsize=8, color="#555555", ha="left", va="center"
        )

    for i, info in enumerate(layers_info[:max_blocks_for_render]):
        b_color = color_map.get(info["type"], "#9B9B9B")
        
        # Rectangular block for layer
        rect = patches.FancyBboxPatch(
            (x_offset, 1.5), box_width, box_height,
            boxstyle="round,pad=0.2",
            ec="#333333", fc=b_color, alpha=0.85, lw=1.5
        )
        ax.add_patch(rect)

        # Text labels inside block
        ax.text(
            x_offset + box_width / 2, 4.4, info["type"],
            fontsize=10, fontweight="bold", ha="center", va="center", color="#000000"
        )
        ax.text(
            x_offset + box_width / 2, 3.7, f"'{info['name']}'",
            fontsize=8, ha="center", va="center", color="#222222", style="italic"
        )
        ax.text(
            x_offset + box_width / 2, 2.9, f"Shape:\n{info['shape']}",
            fontsize=8, ha="center", va="center", color="#111111"
        )
        ax.text(
            x_offset + box_width / 2, 2.0, f"Params: {info['params']:,}",
            fontsize=8, fontweight="bold", ha="center", va="center", color="#000000"
        )

        # Arrow connector to next block
        if i < num_blocks - 1:
            arrow_start_x = x_offset + box_width + 0.1
            arrow_end_x = x_offset + 2.2 - 0.1
            ax.annotate(
                "", xy=(arrow_end_x, 3.25), xytext=(arrow_start_x, 3.25),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#333333")
            )

        x_offset += 2.2

    fig.tight_layout(pad=0.5)
    plt.savefig(svg_path, format="svg")
    plt.savefig(png_path, format="png", dpi=150)
    plt.close(fig)

    logger.info(f"Generated Matplotlib diagram: SVG='{svg_path}', PNG='{png_path}'")


def generate_model_architecture_diagram(
    model: keras.Model,
    output_dir: Path,
    name_prefix: str,
) -> Tuple[Path, Path]:
    """Generate architecture diagram in SVG and PNG formats.

    Tries keras.utils.plot_model first (if pydot/graphviz present), falling back
    gracefully to matplotlib custom block diagram generator.

    Args:
        model: Keras Model instance.
        output_dir: Output directory path.
        name_prefix: File name prefix (e.g. 'custom_cnn').

    Returns:
        Tuple of (svg_path, png_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / f"{name_prefix}_architecture.svg"
    png_path = output_dir / f"{name_prefix}_architecture.png"

    plot_success = False

    # Attempt Keras plot_model with Graphviz/pydot
    try:
        keras.utils.plot_model(
            model,
            to_file=str(png_path),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
        )
        # Convert PNG to SVG or generate SVG via plot_model if supported
        try:
            keras.utils.plot_model(
                model,
                to_file=str(svg_path),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=False,
            )
        except Exception:
            pass
        
        if png_path.exists() and png_path.stat().st_size > 0:
            plot_success = True
            logger.info(f"Generated Keras plot_model diagram at: {png_path}")
    except Exception as err:
        logger.debug(f"Keras plot_model unavailable ({err}). Using fallback matplotlib renderer.")

    # Matplotlib fallback generator
    if not plot_success or not svg_path.exists():
        _render_matplotlib_diagram(model, svg_path, png_path)

    return svg_path, png_path


def visualize_architecture(
    architecture: str = "custom_cnn",
    output_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build model and generate summary TXT, SVG, and PNG visualization artifacts.

    Args:
        architecture: Target architecture ('custom_cnn', 'densenet121', 'efficientnetb0').
        output_dir: Target output directory (defaults to artifacts/architecture).
        config: Master configuration dictionary.

    Returns:
        Dictionary containing summary stats and generated artifact file paths.
    """
    if output_dir is None:
        output_dir = get_project_root() / "artifacts" / "architecture"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model using factory without GPU requirement
    model = build_model(
        architecture=architecture,
        input_shape=(224, 224, 3),
        num_classes=1,
        compile_model=True,
        mixed_precision=False,
        config=config,
    )

    txt_path = output_dir / f"{architecture}_summary.txt"
    summary_txt = generate_model_summary_txt(model, txt_path)

    svg_path, png_path = generate_model_architecture_diagram(model, output_dir, name_prefix=architecture)

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    result = {
        "architecture": architecture,
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "summary_txt_path": txt_path,
        "svg_path": svg_path,
        "png_path": png_path,
    }

    logger.info(
        f"Visualization complete for [{architecture}]: "
        f"Params={total_params:,} | TXT='{txt_path.name}' | SVG='{svg_path.name}' | PNG='{png_path.name}'"
    )

    return result
