"""Script to fetch authentic DICOM files from pydicom built-in medical datasets and open repositories."""

import os
from pathlib import Path
import shutil
import pydicom
from pydicom.data import get_testdata_file
from medvision.config.settings import get_project_root
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.download_samples")

# 10 Authentic DICOM files available in pydicom medical dataset repository
REAL_DICOM_NAMES = [
    "20069792_wlm.dcm",
    "JPEG2000.dcm",
    "CT_small.dcm",
    "MR_small.dcm",
    "rtdose.dcm",
    "OB731570.dcm",
    "reportsi.dcm",
    "JPEG-lossless.dcm",
    "explicit_VR-implicit_meta.dcm",
    "SC_rgb.dcm",
]


def load_authentic_sample_dicoms() -> Path:
    """Copy authentic DICOM files to local data/raw/real_samples directory."""
    root = get_project_root()
    sample_dir = root / "data" / "raw" / "real_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 10 authentic DICOM files to: {sample_dir}")

    for idx, dcm_name in enumerate(REAL_DICOM_NAMES):
        src_path = get_testdata_file(dcm_name)
        dest_filename = f"real_patient_{idx+1:02d}_{dcm_name}"
        dest_path = sample_dir / dest_filename

        if src_path and os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            ds = pydicom.dcmread(dest_path)
            patient_id = getattr(ds, "PatientID", f"PATIENT_{idx+1:02d}")
            modality = getattr(ds, "Modality", "CR")
            shape = ds.pixel_array.shape if hasattr(ds, "pixel_array") else (0, 0)
            print(f"Loaded {dest_filename}: PatientID={patient_id}, Modality={modality}, Dimensions={shape}")

    return sample_dir


if __name__ == "__main__":
    load_authentic_sample_dicoms()
