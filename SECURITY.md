# Security Policy & Guidelines

## Supported Versions

Only the latest `main` branch version of **MedVision-AI** is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability or sensitive data exposure within MedVision-AI:

1. **Do NOT open a public GitHub issue.**
2. Send a detailed report to `security@medvision.ai` or contact the repository maintainers directly.
3. Include the following details in your report:
   - Type of vulnerability (e.g., buffer overflow, secret exposure, dependency vulnerability).
   - Step-by-step instructions to reproduce the issue.
   - Potential impact of the vulnerability.

## Data & Secret Handling Guidelines

- **Zero Datasets in Git**: Medical imaging datasets (DICOM, PNG, CSV manifests) must **never** be committed to the git repository. Ensure dataset paths are listed in `.gitignore`.
- **Zero Hardcoded Credentials**: API secrets, tokens, or environment-specific paths must be managed via `.env` files or environment variables.
- **Model Checkpoints**: Binary model files (`.h5`, `.keras`, `.onnx`) must not be committed to version control; store checkpoints in external cloud storage (e.g., S3, GCS, or Release Artifacts).
