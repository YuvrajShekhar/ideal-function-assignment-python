"""
Ideal Function Fitting and Test Data Mapping
Course: DLMDSPWP01 - Programming with Python

Main implementation module for local execution.
Reads data from data/ folder and outputs to results/ folder.

Author: Yuvraj Shekhar
Matriculation Number: 10244366
Date: 06/11/2025
"""

import pandas as pd
import numpy as np
import sqlite3
import math
import logging
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Database and ORM
from sqlalchemy import create_engine, Column, Integer, Float, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Custom Exception Classes
# =============================================================================

class DataLoadError(Exception):
    """Exception raised when data loading operations fail."""
    pass


class FittingError(Exception):
    """Exception raised when function fitting operations fail."""
    pass


class MappingError(Exception):
    """Exception raised when test data mapping operations fail."""
    pass


# =============================================================================
# DataManager Class
# =============================================================================

class DataManager:
    """
    Manages data loading, database operations, and data persistence.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        engine: SQLAlchemy database engine
        metadata: SQLAlchemy metadata object
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize DataManager with database connection.
        
        Args:
            db_path: Path to SQLite database file. Defaults to results/assignment_data.db
        """
        if db_path is None:
            db_path = str(RESULTS_DIR / "assignment_data.db")
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.metadata = MetaData()
        self.setup_database()
        
    def setup_database(self) -> None:
        """Create database tables if they don't exist."""
        try:
            # Training data table
            self.training_table = Table(
                'training_data', self.metadata,
                Column('x', Float, primary_key=True),
                Column('y1', Float),
                Column('y2', Float),
                Column('y3', Float),
                Column('y4', Float)
            )
            
            # Ideal functions table
            columns = [Column('x', Float, primary_key=True)]
            for i in range(1, 51):
                columns.append(Column(f'y{i}', Float))
            self.ideal_table = Table('ideal_functions', self.metadata, *columns)
            
            # Test results table
            self.test_results_table = Table(
                'test_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('x', Float),
                Column('y', Float),
                Column('delta_y', Float),
                Column('ideal_func_number', Integer)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            raise DataLoadError(f"Database setup failed: {str(e)}")
    
    def load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from CSV files in the data/ directory.
        
        Returns:
            Tuple containing training, test, and ideal dataframes
            
        Raises:
            DataLoadError: If file loading fails
        """
        try:
            train_path = DATA_DIR / "train.csv"
            test_path = DATA_DIR / "test.csv"
            ideal_path = DATA_DIR / "ideal.csv"
            
            # Check if files exist
            for path in [train_path, test_path, ideal_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
            
            # Load dataframes
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            ideal_data = pd.read_csv(ideal_path)
            
            logger.info(f"Loaded training data: {train_data.shape}")
            logger.info(f"Loaded test data: {test_data.shape}")
            logger.info(f"Loaded ideal data: {ideal_data.shape}")
            
            return train_data, test_data, ideal_data
            
        except Exception as e:
            raise DataLoadError(f"CSV loading failed: {str(e)}")
    
    def save_to_database(self, train_data: pd.DataFrame, ideal_data: pd.DataFrame) -> None:
        """
        Save training and ideal data to database.
        
        Args:
            train_data: Training functions dataframe
            ideal_data: Ideal functions dataframe
        """
        try:
            train_data.to_sql('training_data', self.engine, if_exists='replace', index=False)
            ideal_data.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)
            logger.info("Data saved to database successfully")
        except Exception as e:
            raise DataLoadError(f"Database save failed: {str(e)}")
    
    def save_test_results(self, results: List[Dict]) -> None:
        """
        Save test mapping results to database.
        
        Args:
            results: List of dictionaries containing mapping results
        """
        try:
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_sql('test_results', self.engine, if_exists='replace', index=False)
                logger.info(f"Saved {len(results)} test results to database")
        except Exception as e:
            raise DataLoadError(f"Test results save failed: {str(e)}")


# =============================================================================
# FunctionFitter Class
# =============================================================================

class FunctionFitter:
    """
    Performs least squares fitting to find optimal ideal functions.
    
    Attributes:
        train_data: Training functions dataframe
        ideal_data: Ideal functions dataframe
        best_fits: Dictionary of best fit results
    """
    
    def __init__(self, train_data: pd.DataFrame, ideal_data: pd.DataFrame):
        """Initialize FunctionFitter with training and ideal data."""
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.best_fits = {}
    
    def calculate_least_squares_deviation(self, train_y: pd.Series, ideal_y: pd.Series) -> float:
        """
        Calculate sum of squared deviations.
        
        Args:
            train_y: Training function y-values
            ideal_y: Ideal function y-values
            
        Returns:
            Sum of squared deviations
        """
        try:
            deviations = train_y - ideal_y
            return np.sum(deviations ** 2)
        except Exception as e:
            raise FittingError(f"Least squares calculation failed: {str(e)}")
    
    def find_best_fits(self) -> Dict[str, Dict]:
        """
        Find the best fitting ideal function for each training function.
        
        Returns:
            Dictionary mapping training functions to best ideal functions
        """
        try:
            training_functions = ['y1', 'y2', 'y3', 'y4']
            ideal_functions = [f'y{i}' for i in range(1, 51)]
            
            for train_func in training_functions:
                best_deviation = float('inf')
                best_ideal_func = None
                best_ideal_index = None
                
                for i, ideal_func in enumerate(ideal_functions, 1):
                    deviation = self.calculate_least_squares_deviation(
                        self.train_data[train_func], 
                        self.ideal_data[ideal_func]
                    )
                    
                    if deviation < best_deviation:
                        best_deviation = deviation
                        best_ideal_func = ideal_func
                        best_ideal_index = i
                
                self.best_fits[train_func] = {
                    'ideal_function': best_ideal_func,
                    'ideal_index': best_ideal_index,
                    'deviation': best_deviation
                }
                
                logger.info(f"{train_func} → {best_ideal_func} (deviation: {best_deviation:.4f})")
            
            return self.best_fits
            
        except Exception as e:
            raise FittingError(f"Best fit calculation failed: {str(e)}")
    
    def get_max_deviations(self) -> Dict[str, float]:
        """
        Calculate maximum deviations for each training function fit.
        
        Returns:
            Dictionary of maximum deviations per training function
        """
        try:
            max_deviations = {}
            
            for train_func, fit_info in self.best_fits.items():
                ideal_func = fit_info['ideal_function']
                deviations = abs(self.train_data[train_func] - self.ideal_data[ideal_func])
                max_deviations[train_func] = deviations.max()
            
            return max_deviations
            
        except Exception as e:
            raise FittingError(f"Max deviation calculation failed: {str(e)}")


# =============================================================================
# TestMapper Class
# =============================================================================

class TestMapper:
    """
    Maps test data points to ideal functions using sqrt(2) criterion.
    
    Attributes:
        test_data: Test data points
        ideal_data: Ideal functions data
        best_fits: Selected ideal functions
        max_deviations: Maximum training deviations
        sqrt_2: Square root of 2 constant
    """
    
    def __init__(self, test_data: pd.DataFrame, ideal_data: pd.DataFrame,
                 best_fits: Dict, max_deviations: Dict):
        """Initialize TestMapper with necessary data."""
        self.test_data = test_data
        self.ideal_data = ideal_data
        self.best_fits = best_fits
        self.max_deviations = max_deviations
        self.sqrt_2 = math.sqrt(2)
    
    def map_test_point(self, x_test: float, y_test: float) -> Optional[Dict]:
        """
        Attempt to map a single test point to an ideal function.
        
        Args:
            x_test: Test point x-coordinate
            y_test: Test point y-coordinate
            
        Returns:
            Mapping dictionary if successful, None otherwise
        """
        try:
            x_diff = abs(self.ideal_data['x'] - x_test)
            closest_idx = x_diff.idxmin()
            
            best_mapping = None
            min_deviation = float('inf')
            
            for train_func, fit_info in self.best_fits.items():
                ideal_func = fit_info['ideal_function']
                ideal_index = fit_info['ideal_index']
                
                ideal_y = self.ideal_data.loc[closest_idx, ideal_func]
                deviation = abs(y_test - ideal_y)
                
                max_allowed_deviation = self.max_deviations[train_func] * self.sqrt_2
                
                if deviation <= max_allowed_deviation and deviation < min_deviation:
                    min_deviation = deviation
                    best_mapping = {
                        'x': x_test,
                        'y': y_test,
                        'delta_y': deviation,
                        'ideal_func_number': ideal_index
                    }
            
            return best_mapping
            
        except Exception as e:
            raise MappingError(f"Test point mapping failed: {str(e)}")
    
    def map_all_test_data(self) -> List[Dict]:
        """
        Map all test data points to ideal functions.
        
        Returns:
            List of successful mappings
        """
        try:
            mappings = []
            
            for _, row in self.test_data.iterrows():
                mapping = self.map_test_point(row['x'], row['y'])
                if mapping:
                    mappings.append(mapping)
            
            success_rate = (len(mappings) / len(self.test_data)) * 100
            logger.info(f"Mapped {len(mappings)}/{len(self.test_data)} points ({success_rate:.1f}%)")
            
            return mappings
            
        except Exception as e:
            raise MappingError(f"Bulk test mapping failed: {str(e)}")


# =============================================================================
# Visualizer Class
# =============================================================================

class Visualizer:
    """
    Creates comprehensive visualizations for analysis results.
    
    Attributes:
        train_data: Training functions
        ideal_data: Ideal functions
        test_data: Test points
        best_fits: Selected ideal functions
        mappings: Test mapping results
    """
    
    def __init__(self, train_data: pd.DataFrame, ideal_data: pd.DataFrame,
                 test_data: pd.DataFrame, best_fits: Dict, mappings: List[Dict]):
        """Initialize Visualizer with all necessary data."""
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.test_data = test_data
        self.best_fits = best_fits
        self.mappings = mappings
    
    def create_comprehensive_dashboard(self, save_path: str = None):
        """
        Create 4-panel Matplotlib dashboard.
        
        Args:
            save_path: Path to save the figure. If None, saves to results/visualizations/
        """
        if save_path is None:
            save_path = str(VISUALIZATIONS_DIR / "complete_dashboard.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ideal Function Fitting and Test Data Mapping - Complete Analysis', 
                     fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Panel 1: Training vs Ideal Functions
        ax1 = axes[0, 0]
        for train_func, color in zip(['y1', 'y2', 'y3', 'y4'], colors):
            ax1.plot(self.train_data['x'], self.train_data[train_func],
                    color=color, linewidth=2, label=f'Training {train_func}')
            
            ideal_func = self.best_fits[train_func]['ideal_function']
            ax1.plot(self.ideal_data['x'], self.ideal_data[ideal_func],
                    color=color, linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Ideal {ideal_func}')
        
        ax1.set_title('Training Data vs Best-Fit Ideal Functions', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Test Data Mapping
        ax2 = axes[0, 1]
        if self.mappings:
            mapping_df = pd.DataFrame(self.mappings)
            ax2.scatter(mapping_df['x'], mapping_df['y'],
                       c='red', s=50, alpha=0.7, label='Mapped Points')
            
            mapped_indices = set()
            for mapping in self.mappings:
                mask = (abs(self.test_data['x'] - mapping['x']) < 1e-10) & \
                       (abs(self.test_data['y'] - mapping['y']) < 1e-10)
                if mask.any():
                    mapped_indices.add(mask.idxmax())
            
            unmapped_data = self.test_data.drop(list(mapped_indices))
            if not unmapped_data.empty:
                ax2.scatter(unmapped_data['x'], unmapped_data['y'],
                           c='gray', s=50, alpha=0.7, label='Unmapped Points')
        
        ax2.set_title('Test Data Mapping Results', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Deviation Analysis
        ax3 = axes[1, 0]
        if self.mappings:
            mapping_df = pd.DataFrame(self.mappings)
            ax3.scatter(mapping_df['x'], mapping_df['delta_y'],
                       c='blue', s=50, alpha=0.7)
            mean_dev = mapping_df['delta_y'].mean()
            ax3.axhline(y=mean_dev, color='red', linestyle='--',
                       label=f'Mean: {mean_dev:.4f}')
        
        ax3.set_title('Deviation Analysis', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Deviation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Deviation Distribution
        ax4 = axes[1, 1]
        if self.mappings:
            mapping_df = pd.DataFrame(self.mappings)
            ax4.hist(mapping_df['delta_y'], bins=20, alpha=0.7,
                    color='skyblue', edgecolor='black')
            ax4.axvline(mapping_df['delta_y'].mean(), color='red',
                       linestyle='--', label=f"Mean: {mapping_df['delta_y'].mean():.4f}")
            ax4.axvline(mapping_df['delta_y'].median(), color='green',
                       linestyle='--', label=f"Median: {mapping_df['delta_y'].median():.4f}")
        
        ax4.set_title('Distribution of Deviations', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Deviation')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
        plt.show()


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_analysis():
    """
    Execute the complete analysis pipeline.
    
    Returns:
        Dictionary containing all analysis results
    """
    print("\n" + "=" * 70)
    print("IDEAL FUNCTION FITTING AND TEST DATA MAPPING ANALYSIS")
    print("=" * 70)
    
    try:
        # Step 1: Initialize Data Manager
        print("\n[1/7] Initializing Data Manager...")
        data_manager = DataManager()
        
        # Step 2: Load Data
        print("[2/7] Loading CSV Data from data/ directory...")
        train_data, test_data, ideal_data = data_manager.load_csv_data()
        
        # Step 3: Save to Database
        print("[3/7] Saving to Database...")
        data_manager.save_to_database(train_data, ideal_data)
        
        # Step 4: Find Best Fits
        print("[4/7] Finding Best-Fit Ideal Functions...")
        fitter = FunctionFitter(train_data, ideal_data)
        best_fits = fitter.find_best_fits()
        max_deviations = fitter.get_max_deviations()
        
        # Step 5: Map Test Data
        print("[5/7] Mapping Test Data...")
        mapper = TestMapper(test_data, ideal_data, best_fits, max_deviations)
        mappings = mapper.map_all_test_data()
        
        # Step 6: Save Results
        print("[6/7] Saving Results to Database...")
        data_manager.save_test_results(mappings)
        
        # Step 7: Create Visualizations
        print("[7/7] Generating Visualizations...")
        visualizer = Visualizer(train_data, ideal_data, test_data, best_fits, mappings)
        visualizer.create_comprehensive_dashboard()
        
        # Display Summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Test data points: {len(test_data)}")
        print(f"Successfully mapped points: {len(mappings)}")
        print(f"Mapping success rate: {len(mappings)/len(test_data)*100:.1f}%")
        
        print(f"\nBest Fitting Functions:")
        for train_func, fit_info in best_fits.items():
            print(f"  {train_func} → {fit_info['ideal_function']} "
                  f"(deviation: {fit_info['deviation']:.4f})")
        
        if mappings:
            mapping_df = pd.DataFrame(mappings)
            print(f"\nMapping Statistics:")
            print(f"  Mean deviation: {mapping_df['delta_y'].mean():.4f}")
            print(f"  Median deviation: {mapping_df['delta_y'].median():.4f}")
            print(f"  Max deviation: {mapping_df['delta_y'].max():.4f}")
            print(f"  Min deviation: {mapping_df['delta_y'].min():.4f}")
        
        print("\n" + "=" * 70)
        print("✓ Analysis completed successfully!")
        print(f"✓ Database saved to: {data_manager.db_path}")
        print(f"✓ Visualizations saved to: {VISUALIZATIONS_DIR}")
        print("=" * 70)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'ideal_data': ideal_data,
            'best_fits': best_fits,
            'mappings': mappings,
            'max_deviations': max_deviations
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_analysis()