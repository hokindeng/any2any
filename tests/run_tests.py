#!/usr/bin/env python3
"""
Test runner for the glb2glb package.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests():
    """Run all unit tests."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_module(module_name):
    """Run tests for a specific module."""
    loader = unittest.TestLoader()
    
    try:
        module = __import__(f'tests.test_{module_name}', fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Error: Could not find test module 'test_{module_name}'")
        print(f"Available modules: pipeline, importer, exporter, transfer, retargeter")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific module tests
        module = sys.argv[1]
        success = run_specific_module(module)
    else:
        # Run all tests
        print("Running all tests...")
        print("="*60)
        success = run_all_tests()
        
    if success:
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed!")
        print("="*60)
        
    sys.exit(0 if success else 1)
