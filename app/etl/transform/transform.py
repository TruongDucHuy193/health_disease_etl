import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configuration parameters
SELECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'oldpeak', 'slope', 'ca', 'thal', 'dig', 'prop', 'nitr', 'pro',
    'diuretic', 'restef', 'restwm', 'exeref', 'exerwm', 'smoke', 'cigs',
    'htn', 'dm', 'famhist', 'exang', 'xhypo', 'lmt', 'ladprox', 'laddist',
    'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist'
]
CRITICAL_COLUMNS = ['age', 'sex', 'cp']
DEFAULT_VALUES = {
    'slope': 0, 'ca': 0, 'thal': 3, 'restef': 0, 'restwm': 0,
    'exeref': 0, 'exerwm': 0, 'dm': 0, 'famhist': 0, 'lmt': 0,
    'ladprox': 0, 'laddist': 0, 'diag': 0, 'cxmain': 0, 'ramus': 0,
    'om1': 0, 'om2': 0, 'rcaprox': 0, 'rcadist': 0
}
VALID_RANGES = {
    'age': (0, 120),
    'trestbps': (50, 250),
    'chol': (100, 600),
    'thalach': (50, 250),
    'oldpeak': (0, 10)
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    #Convert all column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    return df


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Keep only the specified columns if they exist in the DataFrame."""
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns]


def clean_missing_values(
    df: pd.DataFrame,
    critical_cols: List[str],
    default_vals: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Handle missing values by removing rows missing critical fields and filling others"""
    stats = {
        "rows_with_missing_removed": 0,
        "missing_values_filled": 0
    }

    # Replace common placeholders with NaN
    df = df.replace(['-9', -9, 'NULL', 'null', 'NA', 'N/A', ''], np.nan)

    # Drop rows missing any critical value
    critical_cols_exist = [col for col in critical_cols if col in df.columns]
    if critical_cols_exist:
        missing_critical = df[critical_cols_exist].isna().any(axis=1)
        if missing_critical.any():
            before_rows = len(df)
            df = df[~missing_critical]
            stats["rows_with_missing_removed"] += before_rows - len(df)

    # Drop entirely empty rows
    empty_rows = df.isna().all(axis=1)
    if empty_rows.any():
        before_rows = len(df)
        df = df[~empty_rows]
        stats["rows_with_missing_removed"] += before_rows - len(df)

    missing_before = df.isna().sum().sum()

    for col in df.columns:
        if df[col].isna().any():
            if col in default_vals:
                df[col] = df[col].fillna(default_vals[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                if pd.notna(median_value):
                    df[col] = df[col].fillna(median_value)
            else:
                if not df[col].mode().empty:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)

    missing_after = df.isna().sum().sum()
    stats["missing_values_filled"] = missing_before - missing_after

    return df, stats


def validate_data(
    df: pd.DataFrame,
    valid_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Correct values outside specified numeric ranges by replacing them with the median"""
    stats = {"invalid_values_corrected": 0}

    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                valid_median = df.loc[~invalid_mask, col].median()
                df.loc[invalid_mask, col] = valid_median
                stats["invalid_values_corrected"] += invalid_count

    return df, stats


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop duplicate rows"""
    before_dedup = len(df)
    df = df.drop_duplicates()
    return df, {"duplicates_removed": before_dedup - len(df)}

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Clean the raw DataFrame by removing rows with text values in numeric columns,
    handling extreme values, and removing rows with unrealistic ages'''
    # Original row count for logging
    original_count = len(df)
    removed_count = 0
    
    # Check for text values in columns  should be numeric
    for col in SELECTED_COLUMNS:
        if col in df.columns:
            # Try to convert to numeric, marking text values as NaN
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            text_values = numeric_values.isna() & df[col].notna()
            
            if text_values.any():
                # Log rows will be removed
                print(f"Removing {text_values.sum()} rows with text values in column '{col}'")
                example_values = df.loc[text_values, col].head(3).tolist()
                print(f"  Example values: {example_values}")
                
                # Remove rows
                df = df[~text_values]
                removed_count += text_values.sum()
    
    # Handle extremely large values in numeric columns (like 8223.0, 16781.0)
    for col in df.select_dtypes(include=['number']).columns:
        # Skip column ID and columns where large values might be legitimate
        if col != df.columns[0] and col not in ['id', 'ccf']:
            # Calculate threshold as 99th percentile times 1.5
            valid_values = df[col].dropna()
            if not valid_values.empty:
                percentile_99 = np.percentile(valid_values, 99)
                threshold = percentile_99 * 1.5
                if threshold > 0:
                    extreme_values = df[col] > threshold
                    if extreme_values.any():
                        print(f"Removed {extreme_values.sum()} rows with extreme values in {col} (>{threshold:.1f})")
                        df = df[~extreme_values]
                        removed_count += extreme_values.sum()
    
    # Drop rows with age out of realistic range (0-120)
    if 'age' in df.columns:
        invalid_age = (df['age'] < 0) | (df['age'] > 120)
        if invalid_age.any():
            df = df[~invalid_age]
            removed_count += invalid_age.sum()
            print(f"Removed {invalid_age.sum()} rows with invalid age values")
    
    print(f"Total: Removed {removed_count} corrupted rows out of {original_count}")
    return df


def transform_single_file(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply all cleaning and transformation steps to a single DataFrame."""
    rows_before = len(df)
    stats = {
        "rows_before": rows_before,
        "rows_after": 0,
        "rows_with_missing_removed": 0,
        "duplicates_removed": 0,
        "missing_values_filled": 0,
        "invalid_values_corrected": 0
    }

    df = standardize_column_names(df)
    df = select_columns(df, SELECTED_COLUMNS)

    df, missing_stats = clean_missing_values(df, CRITICAL_COLUMNS, DEFAULT_VALUES)
    stats["rows_with_missing_removed"] += missing_stats["rows_with_missing_removed"]
    stats["missing_values_filled"] += missing_stats["missing_values_filled"]

    df, validation_stats = validate_data(df, VALID_RANGES)
    stats["invalid_values_corrected"] += validation_stats["invalid_values_corrected"]

    df, duplicate_stats = remove_duplicates(df)
    stats["duplicates_removed"] += duplicate_stats["duplicates_removed"]

    stats["rows_after"] = len(df)
    return df, stats


def transform_heart_disease_data(
    input_directory: str = 'app/data/raw/to_csv',
    output_directory: str = 'app/data/processed'
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_directory)
    csv_files = list(input_path.glob('*.csv'))
    if not csv_files:
        return {}, {"error": "No files found"}

    total_stats = {
        "processed_files": 0,
        "rows_before": 0,
        "rows_after": 0,
        "rows_with_missing_removed": 0,
        "duplicates_removed": 0,
        "missing_values_filled": 0,
        "invalid_values_corrected": 0
    }
    
    # Dictionary to store transformed DataFrames
    transformed_dataframes = {}

    for file_path in csv_files:
        file_name = file_path.name
        try:
            print(f"Processing {file_name}...")
            df = pd.read_csv(file_path)
            print(f"Read {len(df)} rows from {file_name}")
            
            # Clean raw data first to remove corrupted rows
            df = clean_raw_data(df)
            print(f"After cleaning: {len(df)} rows remaining")
            
            # Apply transformation steps
            transformed_df, file_stats = transform_single_file(df)

            # Update statistics
            for key in total_stats:
                if key in file_stats:
                    total_stats[key] += file_stats[key]
            total_stats["processed_files"] += 1
            
            # Store transformed DataFrame with its output filename
            output_filename = f"processed_{file_name}"
            transformed_dataframes[output_filename] = transformed_df
            print(f"Transformed {file_name} - now has {len(transformed_df)} rows")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

    # Add source info to the statistics
    total_stats["input_directory"] = str(input_directory)
    total_stats["output_directory"] = str(output_directory)
    total_stats["processed_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return transformed_dataframes, total_stats