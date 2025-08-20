# GLB2GLB Test Suite

Comprehensive unit tests for the glb2glb package.

## Running Tests

### Run all tests:
```bash
python tests/run_tests.py
```

### Run specific module tests:
```bash
python tests/run_tests.py pipeline
python tests/run_tests.py importer
python tests/run_tests.py exporter
python tests/run_tests.py transfer
python tests/run_tests.py retargeter
python tests/run_tests.py integration
```

### Using unittest directly:
```bash
python -m unittest discover tests
```

### Run a specific test class:
```bash
python -m unittest tests.test_pipeline.TestTransferAnimation
```

### Run with coverage:
```bash
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # Creates htmlcov/index.html
```

## Test Structure

- `test_pipeline.py` - Tests for the main motion transfer pipeline
- `test_importer.py` - Tests for GLB to MuJoCo import functionality
- `test_exporter.py` - Tests for MuJoCo to GLB export and NPY animation export
- `test_transfer.py` - Tests for NPY to GLB animation transfer
- `test_retargeter.py` - Tests for IK-based motion retargeting
- `test_integration.py` - Integration tests for the complete pipeline

## Dependencies

Required for all tests:
```bash
pip install numpy
```

Optional (for full test coverage):
```bash
pip install mujoco pygltflib
pip install pink  # For IK retargeting tests
```

## Mocking Strategy

Tests use extensive mocking to avoid dependencies on:
- Actual GLB files
- MuJoCo physics simulation
- File I/O operations
- External libraries (when not installed)

This ensures tests run quickly and reliably in any environment.

## Test Coverage

The test suite covers:
- ✅ Pipeline orchestration
- ✅ GLB import/export
- ✅ Coordinate system transformations
- ✅ Animation data handling
- ✅ NPY format conversion
- ✅ IK-based retargeting (when mink available)
- ✅ Error handling
- ✅ API functions

## Adding New Tests

When adding new functionality:
1. Create corresponding test in the appropriate test file
2. Use mocking to isolate the unit being tested
3. Test both success and failure cases
4. Include integration tests for end-to-end workflows
