# Tests

This directory contains tests for the nodetool-mlx package.

## Integration Tests

Integration tests are located in `tests/integration/` and are designed to run on macOS with Apple Silicon hardware. These tests verify:

- MLX framework availability
- Import of all MLX nodes (Whisper, TTS, Image Generation, Text Generation)
- Availability of required dependencies (mlx-whisper, mflux, mlx-audio)

### Running Integration Tests Locally

To run the integration tests locally on a macOS machine with Apple Silicon:

```bash
# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_mlx_import.py -v
```

### Running on GitHub Actions

Integration tests automatically run on GitHub Actions using macOS 14 (Sonoma) runners with Apple Silicon (M1) for:

- Pull requests to `main` or `develop` branches
- Pushes to `main` or `develop` branches
- Manual workflow dispatch

The workflow is defined in `.github/workflows/integration-tests.yml`.

## Test Markers

Tests are organized using pytest markers:

- `integration`: Tests requiring MLX hardware (automatically applied to all tests in `tests/integration/`)

You can run tests by marker:

```bash
# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```
