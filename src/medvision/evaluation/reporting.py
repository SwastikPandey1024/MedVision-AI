"""Evaluation report visualization and comparison table generation."""

from typing import Dict, Any, List
from pathlib import Path
import json
import pandas as pd
from medvision.utils.logger import get_logger

logger = get_logger("medvision.evaluation.reporting")


def generate_model_comparison_report(
    model_metrics_list: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate comprehensive markdown and JSON comparison table across models.

    CRITICAL CTO RULE:
    Models are ranked PRIMARILY by PR-AUC (Secondary: ROC-AUC). Accuracy is never used
    as the primary ranking metric for imbalanced medical data.

    Args:
        model_metrics_list: List of metric dicts for each evaluated model.
        output_dir: Output directory path.

    Returns:
        Dict containing comparison DataFrame, JSON path, and Markdown path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_comp = pd.DataFrame(model_metrics_list)

    # Sort primarily by PR-AUC descending, secondarily by ROC-AUC descending
    if "pr_auc" in df_comp.columns and "roc_auc" in df_comp.columns:
        df_comp = df_comp.sort_values(by=["pr_auc", "roc_auc"], ascending=[False, False]).reset_index(drop=True)

    md_lines = []
    md_lines.append("# 🏆 MedVision-AI Model Performance Comparison")
    md_lines.append("")
    md_lines.append("> [!IMPORTANT]")
    md_lines.append("> **Ranking Methodology**: Models are ranked **primarily by PR-AUC** and secondarily by ROC-AUC to prioritize pneumonia detection precision under class imbalance. Accuracy is not used for primary ranking.")
    md_lines.append("")
    md_lines.append("## 📊 Model Benchmark Summary Table")
    md_lines.append("")

    headers = ["Rank", "Model Name", "PR-AUC", "ROC-AUC", "Accuracy", "Precision", "Recall / Sens", "Specificity", "F1-Score", "Params"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, row in df_comp.iterrows():
        rank = idx + 1
        m_name = row.get("model_name", "Model")
        pr_auc = f"{row.get('pr_auc', 0.0):.4f}"
        roc_auc = f"{row.get('roc_auc', 0.0):.4f}"
        acc = f"{row.get('accuracy', 0.0):.4f}"
        prec = f"{row.get('precision', 0.0):.4f}"
        rec = f"{row.get('recall_sensitivity', 0.0):.4f}"
        spec = f"{row.get('specificity', 0.0):.4f}"
        f1 = f"{row.get('f1_score', 0.0):.4f}"
        params = f"{row.get('params', 0):,}" if isinstance(row.get('params'), int) else str(row.get('params', 'N/A'))

        md_lines.append(f"| {rank} | **{m_name}** | **{pr_auc}** | {roc_auc} | {acc} | {prec} | {rec} | {spec} | {f1} | {params} |")

    md_report = "\n".join(md_lines)
    md_path = output_dir / "model_comparison_report.md"
    json_path = output_dir / "model_comparison_summary.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_comp.to_dict(orient="records"), f, indent=2)

    logger.info(f"Saved model comparison report to: {md_path}")
    return {
        "comparison_df": df_comp,
        "markdown_path": md_path,
        "json_path": json_path,
    }
