from matplotlib.rcsetup import validate_any
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration parameters
SELECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'oldpeak', 'slope', 'ca', 'thal', 'dig', 'prop', 'nitr', 'pro',
    'diuretic', 'restef', 'restwm', 'exeref', 'exerwm', 'smoke', 'cigs',
    'htn', 'dm', 'famhist', 'exang', 'xhypo', 'lmt', 'ladprox', 'laddist',
    'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist'
]

# Only truly critical columns that are essential for analysis
TRULY_CRITICAL_COLUMNS = ['age', 'sex'] 

# Medical knowledge-based defaults
MEDICAL_DEFAULTS = {
    'trestbps': 120, 'chol': 200, 'thalach': 150, 'fbs': 0, 'restecg': 0,
    'exang': 0, 'oldpeak': 0, 'slope': 1, 'ca': 0, 'thal': 3, 'smoke': 0,
    'cigs': 0, 'htn': 0, 'dm': 0, 'famhist': 0, 'dig': 0, 'prop': 0,
    'nitr': 0, 'pro': 0, 'diuretic': 0, 'restef': 55, 'restwm': 1,
    'exeref': 55, 'exerwm': 1, 'xhypo': 0,
}

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase"""
    df.columns = [col.lower() for col in df.columns]
    return df

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Keep only the specified columns if they exist in the DataFrame"""
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns]

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing value indicators"""
    missing_indicators = [
        '-9', -9, 'NULL', 'null', 'NA', 'N/A', '', ' ', '?', 'nan', 'NaN', 
        'none', 'None', 'NAN', 'Null', 'missing', 'Missing', 'MISSING',
        '.', '..', '...', 'undefined', 'Undefined', '#N/A', '#NULL!', '#DIV/0!'
    ]
    
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(missing_indicators, np.nan)
        # Handle special cases for specific columns
        if col in ['ca', 'thal', 'slope'] and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].replace([0, -9], np.nan)
    
    return df_clean

def handle_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle missing data with medical knowledge and simple imputation"""
    
    stats = {"rows_removed": 0, "values_imputed": 0}
    
    print("🧠 Processing missing data...")
    
    # Clean missing indicators
    df_clean = clean_missing_values(df)
    
    # Remove completely empty rows
    before_empty = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    empty_removed = before_empty - len(df_clean)
    
    # Remove rows missing ALL critical columns
    critical_cols_exist = [col for col in TRULY_CRITICAL_COLUMNS if col in df_clean.columns]
    if critical_cols_exist:
        critical_all_missing = df_clean[critical_cols_exist].isnull().all(axis=1)
        before_critical = len(df_clean)
        df_clean = df_clean[~critical_all_missing]
        critical_removed = before_critical - len(df_clean)
    else:
        critical_removed = 0
    
    stats["rows_removed"] = empty_removed + critical_removed
    
    if stats["rows_removed"] > 0:
        print(f"   🗑️  Removed {stats['rows_removed']} unusable rows")
    
    # Impute missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            missing_count = df_clean[col].isnull().sum()
            
            # Use medical default if available
            if col in MEDICAL_DEFAULTS:
                df_clean[col] = df_clean[col].fillna(MEDICAL_DEFAULTS[col])
                print(f"   🏥 {col}: filled {missing_count} values with medical default")
            # Use median for numeric columns
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"   📊 {col}: filled {missing_count} values with median")
            # Use mode for categorical columns
            else:
                mode_val = df_clean[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(f"   📝 {col}: filled {missing_count} values with mode")
            
            stats["values_imputed"] += missing_count
    
    # Optimize data types
    type_conversions = {
        'sex': 'int8', 'cp': 'int8', 'fbs': 'int8', 'restecg': 'int8',
        'exang': 'int8', 'slope': 'int8', 'ca': 'int8', 'thal': 'int8',
        'smoke': 'int8', 'htn': 'int8', 'dm': 'int8', 'famhist': 'int8',
        'dig': 'int8', 'prop': 'int8', 'nitr': 'int8', 'pro': 'int8',
        'diuretic': 'int8', 'xhypo': 'int8'
    }
    
    for col, dtype in type_conversions.items():
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(dtype)
            except:
                pass
    
    print(f"   ✅ Missing data processed: {stats['values_imputed']} values imputed")
    return df_clean, stats

def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Validate data ranges and fix obvious errors"""
    
    stats = {"invalid_values_corrected": 0}
    
    # Define reasonable ranges for key medical variables
    valid_ranges = {
        'age': (0, 120),
        'trestbps': (50, 300),
        'chol': (50, 800),
        'thalach': (40, 300),
        'oldpeak': (0, 15)
    }
    
    print("🔍 Validating data ranges...")
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                if col in MEDICAL_DEFAULTS:
                    df.loc[invalid_mask, col] = MEDICAL_DEFAULTS[col]
                else:
                    df.loc[invalid_mask, col] = df[col].median()
                
                stats["invalid_values_corrected"] += invalid_count
                print(f"   🔧 {col}: fixed {invalid_count} invalid values")
    
    return df, stats

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate rows"""
    before_count = len(df)
    df_dedup = df.drop_duplicates()
    total_removed = before_count - len(df_dedup)
    
    if total_removed > 0:
        print(f"   🗑️  Removed {total_removed} duplicate rows")
    
    return df_dedup, {"duplicates_removed": total_removed}

def transform_single_file(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform a single file through the complete pipeline"""
    
    rows_before = len(df)
    print(f"🚀 Transforming {rows_before} rows...")
    
    stats = {
        "rows_before": rows_before,
        "rows_after": 0,
        "data_retention_rate": 0,
        "rows_removed": 0,
        "duplicates_removed": 0,
        "values_imputed": 0,
        "invalid_values_corrected": 0
    }

    # Step 1: Basic preprocessing
    df = standardize_column_names(df)
    df = select_columns(df, SELECTED_COLUMNS)
    print(f"   📋 Selected {len(df.columns)} relevant columns")

    # Step 2: Handle missing data
    df, imputation_stats = handle_missing_data(df)
    stats["rows_removed"] += imputation_stats["rows_removed"]
    stats["values_imputed"] += imputation_stats["values_imputed"]

    # Step 3: Validate and clean data
    df, validation_stats = validate_and_clean_data(df)
    stats["invalid_values_corrected"] += validation_stats["invalid_values_corrected"]

    # Step 4: Remove duplicates
    df, duplicate_stats = remove_duplicates(df)
    stats["duplicates_removed"] += duplicate_stats["duplicates_removed"]

    # Final statistics
    stats["rows_after"] = len(df)
    stats["data_retention_rate"] = (len(df) / rows_before) * 100

    print(f"   ✅ Transformation completed: {stats['data_retention_rate']:.1f}% retention")
    
    return df, stats

def transform_heart_disease_data(
    input_directory: str = 'app/data/raw/to_csv',
    output_directory: str = 'app/data/processed'
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Main transformation function for heart disease data"""
    
    print("🏥 HEART DISEASE ETL - DATA TRANSFORMATION")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_directory)
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_directory}")
        return {}, {"error": "No CSV files found in input directory"}

    total_stats = {
        "processed_files": 0,
        "total_rows_before": 0,
        "total_rows_after": 0,
        "overall_retention_rate": 0,
        "total_values_imputed": 0,
        "total_duplicates_removed": 0,
        "total_invalid_corrected": 0
    }
    
    transformed_dataframes = {}

    for file_path in csv_files:
        file_name = file_path.name
        print(f"\n📁 Processing {file_name}...")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Apply transformation
            transformed_df, file_stats = transform_single_file(df)
            
            # Store results
            output_filename = f"processed_{file_name}"
            transformed_dataframes[output_filename] = transformed_df
            
            # Update total statistics
            total_stats["processed_files"] += 1
            total_stats["total_rows_before"] += file_stats["rows_before"]
            total_stats["total_rows_after"] += file_stats["rows_after"]
            total_stats["total_values_imputed"] += file_stats["values_imputed"]
            total_stats["total_duplicates_removed"] += file_stats["duplicates_removed"]
            total_stats["total_invalid_corrected"] += file_stats["invalid_values_corrected"]
            
            print(f"   ✅ {file_name} processed successfully")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_name}: {e}")
            continue

    # Calculate overall statistics
    if total_stats["total_rows_before"] > 0:
        total_stats["overall_retention_rate"] = (
            total_stats["total_rows_after"] / total_stats["total_rows_before"]
        ) * 100
    
    # Summary
    print(f"\n🎉 TRANSFORMATION SUMMARY")
    print(f"=" * 40)
    print(f"📁 Files processed: {total_stats['processed_files']}")
    print(f"📊 Overall data retention: {total_stats['overall_retention_rate']:.1f}%")
    print(f"📈 Total rows: {total_stats['total_rows_before']} → {total_stats['total_rows_after']}")
    print(f"🔧 Total values imputed: {total_stats['total_values_imputed']}")
    print(f"🗑️  Total duplicates removed: {total_stats['total_duplicates_removed']}")
    
    return transformed_dataframes, total_stats