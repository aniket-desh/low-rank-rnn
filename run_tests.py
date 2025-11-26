#!/usr/bin/env python3
"""
Test runner script for lowrank-rnn package.

Usage:
    python run_tests.py ou              # Run test_ou.py
    python run_tests.py all             # Run all tests
    python run_tests.py ou -v           # Run with verbose output
    python run_tests.py ou --no-pytest  # Run test file directly (no pytest)
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

# map test names to test files
TEST_MAP = {
    "ou": "tests/test_ou.py",
    "rnn": "tests/test_rnn.py",
    "low_rank": "tests/test_low_rank.py",
    "all": "tests/",
}

def run_with_pytest(test_file, pytest_args):
    """Run tests using pytest."""
    cmd = ["python", "-m", "pytest", str(test_file)] + pytest_args
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    result = subprocess.run(cmd)
    return result.returncode

def run_directly(test_file):
    """Run test file directly (without pytest)."""
    print(f"Running test file directly: {test_file}")
    print("-" * 60)
    
    # Load and execute the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load test file {test_file}")
        return 1
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run all test functions
    test_functions = [name for name in dir(module) if name.startswith("test_")]
    if not test_functions:
        print("No test functions found (functions starting with 'test_')")
        return 1
    
    print(f"Found {len(test_functions)} test function(s): {', '.join(test_functions)}")
    print()
    
    passed = 0
    failed = 0
    
    for test_name in test_functions:
        test_func = getattr(module, test_name)
        print(f"Running {test_name}...", end=" ")
        try:
            test_func()
            print("✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable tests:")
        for name in TEST_MAP.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    test_name = sys.argv[1].lower()
    
    if test_name not in TEST_MAP:
        print(f"Error: Unknown test '{test_name}'")
        print(f"Available tests: {', '.join(TEST_MAP.keys())}")
        sys.exit(1)
    
    test_path = TEST_MAP[test_name]
    test_file = Path(__file__).parent / test_path
    
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)
    
    # check for --no-pytest flag
    args = sys.argv[2:]
    use_pytest = "--no-pytest" not in args
    if "--no-pytest" in args:
        args.remove("--no-pytest")
    
    if use_pytest:
        # try pytest first
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True,
                check=True
            )
            # pytest is available, use it directly
            returncode = run_with_pytest(test_file, args)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: pytest not found. Running tests directly...")
            print("(Install pytest with: pip install pytest)")
            print()
            returncode = run_directly(test_file)
    else:
        returncode = run_directly(test_file)
    
    sys.exit(returncode)

if __name__ == "__main__":
    main()

