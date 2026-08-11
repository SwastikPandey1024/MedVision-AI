"""Exploratory Data Analysis (EDA) report and dataset statistics generator."""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from medvision.config.settings import get_project_root
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.eda")


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Generate comprehensive EDA metrics and export JSON & Markdown summary reports.

    Args:
        df: Processed RSNA manifest DataFrame.
        output_dir: Output directory to save reports. Defaults to artifacts/experiments/.

    Returns:
        Dict containing EDA metrics summary.
    """
    if output_dir is None:
        output_dir = get_project_root() / "artifacts" / "experiments"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_patients = len(df)
    target_counts = df["target"].value_counts().to_dict()
    pos_count = target_counts.get(1, 0)
    neg_count = target_counts.get(0, 0)
    pos_percent = (pos_count / total_patients * 100) if total_patients > 0 else 0.0

    class_counts = df["detailed_class"].value_counts().to_dict()

    # Bounding box statistics
    positive_df = df[df["target"] == 1]
    total_bboxes = int(positive_df["bbox_count"].sum())
    avg_bboxes_per_pos_patient = float(positive_df["bbox_count"].mean()) if len(positive_df) > 0 else 0.0

    bbox_widths = []
    bbox_heights = []
    for _, row in positive_df.iterrows():
        bboxes = row.get("bboxes", [])
        if isinstance(bboxes, list):
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox_widths.append(bbox[2])
                    bbox_heights.append(bbox[3])

    mean_bbox_width = float(np.mean(bbox_widths)) if bbox_widths else 0.0
    mean_bbox_height = float(np.mean(bbox_heights)) if bbox_heights else 0.0

    report_data: Dict[str, Any] = {
        "dataset_name": "RSNA Pneumonia Detection Challenge",
        "total_unique_patients": total_patients,
        "class_distribution": {
            "negative_normal_count": neg_count,
            "positive_pneumonia_count": pos_count,
            "positive_percentage": round(pos_percent, 2),
        },
        "detailed_class_distribution": class_counts,
        "bounding_box_statistics": {
            "total_bounding_boxes": total_bboxes,
            "mean_bboxes_per_positive_patient": round(avg_bboxes_per_pos_patient, 2),
            "mean_bbox_width_px": round(mean_bbox_width, 2),
            "mean_bbox_height_px": round(mean_bbox_height, 2),
        },
    }

    # Save JSON report
    json_path = output_dir / "eda_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    md_path = output_dir / "eda_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 RSNA Pneumonia Dataset EDA & Quality Report\n\n")
        f.write("## 1. Overview Statistics\n")
        f.write(f"- **Total Unique Patients / Images**: `{total_patients}`\n")
        f.write(f"- **Positive (Pneumonia) Cases**: `{pos_count}` ({pos_percent:.2f}%)\n")
        f.write(f"- **Negative (Normal / No Opacity) Cases**: `{neg_count}` ({100-pos_percent:.2f}%)\n\n")

        f.write("## 2. Detailed Class Breakdown\n")
        for cls, cnt in class_counts.items():
            f.write(f"- **{cls}**: `{cnt}` ({cnt/total_patients*100:.2f}%)\n")

        f.write("\n## 3. Bounding Box Annotation Analysis\n")
        f.write(f"- **Total Bounding Boxes Extracted**: `{total_bboxes}`\n")
        f.write(f"- **Average Bboxes per Positive Patient**: `{avg_bboxes_per_pos_patient:.2f}`\n")
        f.write(f"- **Mean Bbox Dimensions**: `{mean_bbox_width:.1f} x {mean_bbox_height:.1f}` pixels\n")

    logger.info(f"EDA report generated successfully at: {json_path} and {md_path}")
    return report_data
