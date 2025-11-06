"""
Unit Tests for Ideal Function Fitting and Test Data Mapping
Course: DLMDSPWP01 - Programming with Python

Comprehensive test suite for all components.

Author: Yuvraj Shekhar
Matriculation Number: 10244366
Date: 29/07/2025
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import sqlite3
from pathlib import Path

# Import the main module components
from main import (
    DataManager, FunctionFitter, TestMapper, Visualizer,
    DataLoadError, FittingError, MappingError,
    DATA_DIR
)


class TestDataManager(unittest.TestCase):
    """Test DataManager class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test.db")
        self.data_manager = DataManager(self.test_db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_database_initialization(self):
        """Test database is created with correct tables."""
        self.assertTrue(os.path.exists(self.test_db_path))
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['training_data', 'ideal_functions', 'test_results']
        for table in expected_tables:
            self.assertIn(table, tables)
        
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
        
        sample_ideal = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y1': [1.0, 4.0, 9.0],
            'y2': [1.0, 2.0, 3.0]
        })
        
        self.data_manager.save_to_database(sample_train, sample_ideal)
        
        conn = sqlite3.connect(self.test_db_path)
        train_result = pd.read_sql_query("SELECT * FROM training_data", conn)
        ideal_result = pd.read_sql_query("SELECT * FROM ideal_functions", conn)
        
        self.assertEqual(len(train_result), 3)
        self.assertEqual(len(ideal_result), 3)
        
        conn.close()
    
    def test_csv_loading(self):
        """Test CSV file loading from data directory."""
        if (DATA_DIR / "train.csv").exists():
            try:
                train, test, ideal = self.data_manager.load_csv_data()
                
                self.assertIsInstance(train, pd.DataFrame)
                self.assertIsInstance(test, pd.DataFrame)
                self.assertIsInstance(ideal, pd.DataFrame)
                
                self.assertEqual(len(train.columns), 5)
                self.assertEqual(len(test.columns), 2)
                self.assertEqual(len(ideal.columns), 51)
            except Exception as e:
                self.fail(f"CSV loading failed: {e}")


class TestFunctionFitter(unittest.TestCase):
    """Test FunctionFitter class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.train_data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'y1': [1.0, 4.0, 9.0, 16.0],  # x^2
            'y2': [1.0, 2.0, 3.0, 4.0],   # x
            'y3': [2.0, 4.0, 6.0, 8.0],   # 2x
            'y4': [1.0, 8.0, 27.0, 64.0]  # x^3
        })
        
        self.ideal_data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'y1': [1.0, 4.0, 9.0, 16.0],    # Perfect match for train y1
            'y2': [1.0, 2.0, 3.0, 4.0],     # Perfect match for train y2
            'y3': [2.0, 4.0, 6.0, 8.0],     # Perfect match for train y3
            'y4': [1.0, 8.0, 27.0, 64.0],   # Perfect match for train y4
            'y5': [1.5, 2.5, 3.5, 4.5],     # Close to y2 but worse
        })
        
        self.fitter = FunctionFitter(self.train_data, self.ideal_data)
    
    def test_least_squares_calculation(self):
        """Test least squares deviation calculation."""
        # Perfect match should give 0 deviation
        deviation = self.fitter.calculate_least_squares_deviation(
            self.train_data['y1'], self.ideal_data['y1']
        )
        self.assertAlmostEqual(deviation, 0.0, places=10)
        
        # Test with some deviation
        deviation = self.fitter.calculate_least_squares_deviation(
            self.train_data['y2'], self.ideal_data['y5']
        )
        self.assertGreater(deviation, 0)
    
    def test_find_best_fits(self):
        """Test finding best fitting ideal functions."""
        best_fits = self.fitter.find_best_fits()
        
        self.assertEqual(len(best_fits), 4)
        
        # Check perfect matches are found
        self.assertEqual(best_fits['y1']['ideal_function'], 'y1')
        self.assertEqual(best_fits['y2']['ideal_function'], 'y2')
        self.assertEqual(best_fits['y3']['ideal_function'], 'y3')
        self.assertEqual(best_fits['y4']['ideal_function'], 'y4')
        
        # Check deviations are minimal
        for train_func in ['y1', 'y2', 'y3', 'y4']:
            self.assertAlmostEqual(best_fits[train_func]['deviation'], 0.0, places=10)
    
    def test_get_max_deviations(self):
        """Test maximum deviation calculation."""
        self.fitter.find_best_fits()
        max_deviations = self.fitter.get_max_deviations()
        
        self.assertEqual(len(max_deviations), 4)
        
        # For perfect matches, max deviation should be 0
        for train_func in ['y1', 'y2', 'y3', 'y4']:
            self.assertAlmostEqual(max_deviations[train_func], 0.0, places=10)


class TestTestMapper(unittest.TestCase):
    """Test TestMapper class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [1.0, 4.0, 9.0]
        })
        
        self.ideal_data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'y1': [1.0, 4.0, 9.0, 16.0],
            'y2': [1.0, 2.0, 3.0, 4.0]
        })
        
        self.best_fits = {
            'y1': {'ideal_function': 'y1', 'ideal_index': 1, 'deviation': 0.0},
            'y2': {'ideal_function': 'y2', 'ideal_index': 2, 'deviation': 0.0}
        }
        
        self.max_deviations = {'y1': 0.1, 'y2': 0.2}
        
        self.mapper = TestMapper(
            self.test_data, self.ideal_data, self.best_fits, self.max_deviations
        )
    
    def test_map_test_point_success(self):
        """Test successful test point mapping."""
        mapping = self.mapper.map_test_point(1.0, 1.0)
        
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping['x'], 1.0)
        self.assertEqual(mapping['y'], 1.0)
        self.assertEqual(mapping['ideal_func_number'], 1)
        self.assertAlmostEqual(mapping['delta_y'], 0.0, places=10)
    
    def test_map_all_test_data(self):
        """Test mapping all test data points."""
        mappings = self.mapper.map_all_test_data()
        
        # All test points should map successfully (perfect matches)
        self.assertEqual(len(mappings), 3)
        
        # Check first mapping
        self.assertEqual(mappings[0]['x'], 1.0)
        self.assertEqual(mappings[0]['y'], 1.0)


class TestMathematicalOperations(unittest.TestCase):
    """Test mathematical operations and algorithms."""
    
    def test_numpy_vectorization(self):
        """Test NumPy vectorized operations."""
        y1 = pd.Series([1, 2, 3, 4, 5])
        y2 = pd.Series([1, 2, 3, 4, 5])
        
        deviation = np.sum((y1 - y2) ** 2)
        self.assertEqual(deviation, 0)
    
    def test_sqrt_2_criterion(self):
        """Test sqrt(2) criterion calculation."""
        import math
        max_dev = 0.5
        threshold = max_dev * math.sqrt(2)
        
        self.assertAlmostEqual(threshold, 0.7071067811865476, places=10)


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
    
    # Run tests
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
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)

