# Contributing to MedVision-AI

Thank you for contributing to MedVision-AI! This document outlines our development process, coding standards, and submission guidelines.

## Development Workflow

1. **Fork & Clone**: Fork the repository and clone it locally.
2. **Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/phase-X-description
   ```

## Code Quality Standards

Before committing, format and lint your code:

```bash
# Format code
black src tests
isort src tests

# Linting and Type Checks
flake8 src tests
mypy src
```

## Running Tests

Run the Pytest suite to verify your changes:

```bash
pytest tests/
```

## Commit Message Guidelines

We follow Conventional Commits:

- `feat(data)`: Add patient-aware splitter implementation
- `fix(api)`: Fix image decoding error on corrupt JPEG input
- `docs(adr)`: Add ADR-003 for patient splitting decision
- `test(models)`: Add unit tests for DenseNet builder

## Pull Request Process

1. Ensure all tests and linter checks pass.
2. Update documentation and `CHANGELOG.md` accordingly.
3. Open a Pull Request against `main` with a clear summary of changes and validation results.
