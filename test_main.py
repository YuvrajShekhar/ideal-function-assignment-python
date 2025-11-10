"""
Unit Tests for Ideal Function Fitting and Test Data Mapping
Course: DLMDSPWP01 - Programming with Python

Comprehensive test suite for all components.

Author: Yuvraj Shekhar
Matriculation Number: 10244366
Date: 10/11/2025
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import sqlite3
from pathlib import Path
import sys
import shutil

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module components
try:
    from main import (
        DataManager, FunctionFitter, TestMapper, Visualizer,
        DataLoadError, FittingError, MappingError,
        DATA_DIR
    )
except ImportError:
    print("Warning: Could not import main module. Please ensure main.py is in the same directory.")
    sys.exit(1)


class TestDataManager(unittest.TestCase):
    """Test DataManager class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test.db")
        self.data_manager = DataManager(self.test_db_path)
        
        # Create test data directory if needed
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.test_data_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass
        
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_database_initialization(self):
        """Test database is created with correct tables."""
        self.assertTrue(os.path.exists(self.test_db_path))
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['training_data', 'ideal_functions', 'test_results']
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found in database")
        
        conn.close()
    
    def test_save_to_database(self):
        """Test saving data to database."""
        sample_train = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y1': [1.0, 4.0, 9.0],
            'y2': [1.0, 2.0, 3.0],
            'y3': [2.0, 4.0, 6.0],
            'y4': [1.0, 8.0, 27.0]
        })
        
        # Create sample ideal data with all 50 columns
        sample_ideal = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        for i in range(1, 51):
            sample_ideal[f'y{i}'] = [1.0 * i, 2.0 * i, 3.0 * i]
        
        self.data_manager.save_to_database(sample_train, sample_ideal)
        
        conn = sqlite3.connect(self.test_db_path)
        train_result = pd.read_sql_query("SELECT * FROM training_data", conn)
        ideal_result = pd.read_sql_query("SELECT * FROM ideal_functions", conn)
        
        self.assertEqual(len(train_result), 3, "Training data not saved correctly")
        self.assertEqual(len(ideal_result), 3, "Ideal data not saved correctly")
        self.assertEqual(len(train_result.columns), 5, "Training data should have 5 columns")
        self.assertEqual(len(ideal_result.columns), 51, "Ideal data should have 51 columns")
        
        conn.close()
    
    def test_save_test_results(self):
        """Test saving test results to database."""
        test_results = [
            {'x': 1.0, 'y': 1.0, 'delta_y': 0.1, 'ideal_func_number': 1},
            {'x': 2.0, 'y': 4.0, 'delta_y': 0.2, 'ideal_func_number': 2}
        ]
        
        self.data_manager.save_test_results(test_results)
        
        conn = sqlite3.connect(self.test_db_path)
        results = pd.read_sql_query("SELECT * FROM test_results", conn)
        
        self.assertEqual(len(results), 2, "Test results not saved correctly")
        self.assertIn('x', results.columns)
        self.assertIn('y', results.columns)
        self.assertIn('delta_y', results.columns)
        self.assertIn('ideal_func_number', results.columns)
        
        conn.close()


class TestFunctionFitter(unittest.TestCase):
    """Test FunctionFitter class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create more realistic test data
        x_values = np.linspace(-10, 10, 50)
        
        self.train_data = pd.DataFrame({
            'x': x_values,
            'y1': x_values ** 2,           # Quadratic
            'y2': x_values,                 # Linear
            'y3': 2 * x_values,             # Linear with slope 2
            'y4': x_values ** 3 / 100       # Cubic scaled
        })
        
        # Create ideal data that includes perfect matches and variations
        self.ideal_data = pd.DataFrame({'x': x_values})
        
        # Add perfect matches
        self.ideal_data['y1'] = x_values ** 2              # Perfect match for train y1
        self.ideal_data['y2'] = x_values                   # Perfect match for train y2
        self.ideal_data['y3'] = 2 * x_values               # Perfect match for train y3
        self.ideal_data['y4'] = x_values ** 3 / 100        # Perfect match for train y4
        
        # Add variations
        for i in range(5, 51):
            self.ideal_data[f'y{i}'] = x_values ** 2 + np.random.normal(0, 0.1 * i, len(x_values))
        
        self.fitter = FunctionFitter(self.train_data, self.ideal_data)
    
    def test_least_squares_calculation(self):
        """Test least squares deviation calculation."""
        # Perfect match should give 0 deviation
        deviation = self.fitter.calculate_least_squares_deviation(
            self.train_data['y1'], self.ideal_data['y1']
        )
        self.assertAlmostEqual(deviation, 0.0, places=5)
        
        # Test with some deviation
        deviation = self.fitter.calculate_least_squares_deviation(
            self.train_data['y2'], self.ideal_data['y5']
        )
        self.assertGreater(deviation, 0, "Deviation should be positive for non-matching functions")
    
    def test_find_best_fits(self):
        """Test finding best fitting ideal functions."""
        best_fits = self.fitter.find_best_fits()
        
        self.assertEqual(len(best_fits), 4, "Should find 4 best fits")
        
        # Check that each training function has a best fit
        for train_func in ['y1', 'y2', 'y3', 'y4']:
            self.assertIn(train_func, best_fits)
            self.assertIn('ideal_function', best_fits[train_func])
            self.assertIn('deviation', best_fits[train_func])
            self.assertIn('ideal_index', best_fits[train_func])
        
        # For our perfect matches, deviations should be minimal
        for train_func in ['y1', 'y2', 'y3', 'y4']:
            self.assertLessEqual(best_fits[train_func]['deviation'], 1e-5,
                               f"Deviation for {train_func} should be minimal")
    
    def test_get_max_deviations(self):
        """Test maximum deviation calculation."""
        self.fitter.find_best_fits()
        max_deviations = self.fitter.get_max_deviations()
        
        self.assertEqual(len(max_deviations), 4, "Should have 4 max deviations")
        
        # Max deviations should be non-negative
        for train_func in ['y1', 'y2', 'y3', 'y4']:
            self.assertGreaterEqual(max_deviations[train_func], 0,
                                  f"Max deviation for {train_func} should be non-negative")


class TestTestMapper(unittest.TestCase):
    """Test TestMapper class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        x_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        self.test_data = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.25, 2.25, 6.25]  # Approximately x^2
        })
        
        self.ideal_data = pd.DataFrame({
            'x': x_values,
            'y1': x_values ** 2,  # Quadratic
            'y2': x_values,       # Linear
            'y3': 2 * x_values,   # Linear with slope 2
            'y4': x_values ** 3   # Cubic
        })
        
        self.best_fits = {
            'y1': {'ideal_function': 'y1', 'ideal_index': 1, 'deviation': 0.0},
            'y2': {'ideal_function': 'y2', 'ideal_index': 2, 'deviation': 0.0},
            'y3': {'ideal_function': 'y3', 'ideal_index': 3, 'deviation': 0.0},
            'y4': {'ideal_function': 'y4', 'ideal_index': 4, 'deviation': 0.0}
        }
        
        self.max_deviations = {'y1': 0.5, 'y2': 0.3, 'y3': 0.4, 'y4': 0.6}
        
        self.mapper = TestMapper(
            self.test_data, self.ideal_data, self.best_fits, self.max_deviations
        )
    
    def test_map_test_point_success(self):
        """Test successful test point mapping."""
        # Test a point that should map to y1 (quadratic)
        mapping = self.mapper.map_test_point(1.0, 1.0)
        
        self.assertIsNotNone(mapping, "Mapping should not be None")
        if mapping:
            self.assertEqual(mapping['x'], 1.0)
            self.assertEqual(mapping['y'], 1.0)
            self.assertIn('ideal_func_number', mapping)
            self.assertIn('delta_y', mapping)
    
    def test_map_test_point_no_match(self):
        """Test test point that doesn't match any ideal function."""
        # Test a point far from any ideal function
        mapping = self.mapper.map_test_point(1.0, 1000.0)
        
        self.assertIsNone(mapping, "Point far from ideal functions should not map")
    
    def test_map_all_test_data(self):
        """Test mapping all test data points."""
        mappings = self.mapper.map_all_test_data()
        
        # Should return a list
        self.assertIsInstance(mappings, list)
        
        # Check structure of mappings
        for mapping in mappings:
            self.assertIn('x', mapping)
            self.assertIn('y', mapping)
            self.assertIn('ideal_func_number', mapping)
            self.assertIn('delta_y', mapping)


class TestMathematicalOperations(unittest.TestCase):
    """Test mathematical operations and algorithms."""
    
    def test_numpy_vectorization(self):
        """Test NumPy vectorized operations."""
        y1 = pd.Series([1, 2, 3, 4, 5])
        y2 = pd.Series([1, 2, 3, 4, 5])
        
        deviation = np.sum((y1 - y2) ** 2)
        self.assertEqual(deviation, 0, "Identical series should have 0 deviation")
        
        y3 = pd.Series([2, 3, 4, 5, 6])
        deviation = np.sum((y1 - y3) ** 2)
        self.assertEqual(deviation, 5, "Deviation calculation incorrect")
    
    def test_sqrt_2_criterion(self):
        """Test sqrt(2) criterion calculation."""
        import math
        max_dev = 0.5
        threshold = max_dev * math.sqrt(2)
        
        expected = 0.7071067811865476
        self.assertAlmostEqual(threshold, expected, places=10)
    
    def test_interpolation_accuracy(self):
        """Test interpolation for intermediate x-values."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])  # y = x^2
        
        # Test interpolation at x=1.5
        x_interp = 1.5
        y_interp = np.interp(x_interp, x, y)
        
        # For linear interpolation between (1,1) and (2,4)
        expected = 2.5  # Linear interpolation
        self.assertAlmostEqual(y_interp, expected, places=10)


class TestExceptionHandling(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_data_load_error(self):
        """Test DataLoadError exception."""
        with self.assertRaises(DataLoadError):
            raise DataLoadError("Test error message")
    
    def test_fitting_error(self):
        """Test FittingError exception."""
        with self.assertRaises(FittingError):
            raise FittingError("Test fitting error")
    
    def test_mapping_error(self):
        """Test MappingError exception."""
        with self.assertRaises(MappingError):
            raise MappingError("Test mapping error")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test data
        np.random.seed(42)
        x_values = np.linspace(-10, 10, 100)
        
        self.train_data = pd.DataFrame({
            'x': x_values,
            'y1': x_values ** 2 + np.random.normal(0, 0.1, len(x_values)),
            'y2': 2 * x_values + np.random.normal(0, 0.1, len(x_values)),
            'y3': np.sin(x_values) + np.random.normal(0, 0.05, len(x_values)),
            'y4': x_values ** 3 / 50 + np.random.normal(0, 0.1, len(x_values))
        })
        
        # Create ideal functions
        self.ideal_data = pd.DataFrame({'x': x_values})
        self.ideal_data['y1'] = x_values ** 2
        self.ideal_data['y2'] = 2 * x_values
        self.ideal_data['y3'] = np.sin(x_values)
        self.ideal_data['y4'] = x_values ** 3 / 50
        
        # Add more ideal functions
        for i in range(5, 51):
            self.ideal_data[f'y{i}'] = np.random.randn(len(x_values))
        
        # Create test data
        test_x = np.random.uniform(-10, 10, 20)
        self.test_data = pd.DataFrame({
            'x': test_x,
            'y': test_x ** 2 + np.random.normal(0, 0.5, len(test_x))
        })
    
    def tearDown(self):
        """Clean up after integration tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow(self):
        """Test the complete workflow from data loading to mapping."""
        # Initialize components
        db_path = os.path.join(self.temp_dir, "test_integration.db")
        data_manager = DataManager(db_path)
        
        # Save data to database
        data_manager.save_to_database(self.train_data, self.ideal_data)
        
        # Perform fitting
        fitter = FunctionFitter(self.train_data, self.ideal_data)
        best_fits = fitter.find_best_fits()
        max_deviations = fitter.get_max_deviations()
        
        # Verify fitting results
        self.assertEqual(len(best_fits), 4)
        self.assertEqual(len(max_deviations), 4)
        
        # Perform mapping
        mapper = TestMapper(self.test_data, self.ideal_data, best_fits, max_deviations)
        mappings = mapper.map_all_test_data()
        
        # Verify some mappings exist
        self.assertIsInstance(mappings, list)
        if mappings:
            # Save results to database
            data_manager.save_test_results(mappings)
            
            # Verify results were saved
            conn = sqlite3.connect(db_path)
            results = pd.read_sql_query("SELECT * FROM test_results", conn)
            self.assertEqual(len(results), len(mappings))
            conn.close()


def run_tests():
    """Run all tests and display results."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFunctionFitter))
    suite.addTests(loader.loadTestsFromTestCase(TestTestMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestMathematicalOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"Success Rate: 100%")
        print("All tests passed successfully!")
    else:
        success_count = result.testsRun - len(result.failures) - len(result.errors)
        print(f"Success Rate: {(success_count / result.testsRun * 100):.1f}%")
        
        if result.failures:
            print("\nFailed Tests:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
