# API Documentation

## DataManager Class

### `__init__(db_path: str = "assignment_data.db")`
Initialize DataManager with database connection.

**Parameters:**
- `db_path` (str): Path to SQLite database file

**Example:**
```python
dm = DataManager("my_analysis.db")
```

### `load_csv_data(uploaded_files: dict) -> Tuple[DataFrame, DataFrame, DataFrame]`
Load CSV files from uploaded dictionary.

**Parameters:**
- `uploaded_files` (dict): Dictionary of uploaded files

**Returns:**
- Tuple of (training_data, test_data, ideal_data)

**Raises:**
- `DataLoadError`: If file loading fails

### `save_to_database(train_data: DataFrame, ideal_data: DataFrame) -> None`
Save data to SQLite database.

**Parameters:**
- `train_data` (DataFrame): Training functions
- `ideal_data` (DataFrame): Ideal functions

---

## FunctionFitter Class

### `__init__(train_data: DataFrame, ideal_data: DataFrame)`
Initialize with training and ideal function data.

### `calculate_least_squares_deviation(train_y: Series, ideal_y: Series) -> float`
Calculate sum of squared deviations.

**Parameters:**
- `train_y` (Series): Training function values
- `ideal_y` (Series): Ideal function values

**Returns:**
- float: Sum of squared deviations

**Formula:**
```
Σ(train_y[i] - ideal_y[i])²
```

### `find_best_fits() -> Dict[str, Dict]`
Find optimal ideal function for each training function.

**Returns:**
```python
{
    'y1': {'ideal_function': 'y42', 'ideal_index': 42, 'deviation': 34.2466},
    'y2': {'ideal_function': 'y41', 'ideal_index': 41, 'deviation': 35.6018},
    ...
}
```

### `get_max_deviations() -> Dict[str, float]`
Calculate maximum deviations for √2 threshold.

**Returns:**
- Dictionary mapping training functions to max deviations

---

## TestMapper Class

### `__init__(test_data, ideal_data, best_fits, max_deviations)`
Initialize mapper with necessary data.

### `map_test_point(x_test: float, y_test: float) -> Optional[Dict]`
Map single test point to ideal function.

**Parameters:**
- `x_test` (float): X-coordinate
- `y_test` (float): Y-coordinate

**Returns:**
- Dictionary with mapping details or None

**Example Return:**
```python
{
    'x': 1.5,
    'y': 2.25,
    'delta_y': 0.15,
    'ideal_func_number': 42
}
```

### `map_all_test_data() -> List[Dict]`
Map all test points.

**Returns:**
- List of successful mappings

---

## Visualizer Class

### `__init__(train_data, ideal_data, test_data, best_fits, mappings)`
Initialize with all data for visualization.

### `create_comprehensive_dashboard()`
Generate 4-panel Matplotlib dashboard.

**Panels:**
1. Training vs Ideal Functions
2. Test Data Mapping
3. Deviation Analysis
4. Statistical Distribution

### `create_interactive_plots()`
Generate interactive Bokeh visualizations.

**Features:**
- Zoom and pan
- Hover tooltips
- Legend toggling

---

## Usage Example
```python
# Initialize
dm = DataManager()

# Load data
train, test, ideal = dm.load_csv_data(uploaded)

# Find best fits
fitter = FunctionFitter(train, ideal)
best_fits = fitter.find_best_fits()
max_devs = fitter.get_max_deviations()

# Map test data
mapper = TestMapper(test, ideal, best_fits, max_devs)
mappings = mapper.map_all_test_data()

# Visualize
viz = Visualizer(train, ideal, test, best_fits, mappings)
viz.create_comprehensive_dashboard()
viz.create_interactive_plots()

# Save results
dm.save_test_results(mappings)
```