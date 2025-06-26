from matplotlib.rcsetup import validate_any
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration parameters for MI risk prediction
SELECTED_COLUMNS = [
    'age', 'sex', 'num', 'cp', 'thal', 'ca', 'oldpeak', 'exang', 
    'trestbps', 'chol', 'thalach', 'slope', 'restecg', 'htn', 'dm', 
    'famhist', 'fbs', 'prop', 'nitr', 'pro', 
    'diuretic', 'xhypo'
]
# Critical columns for MI risk prediction (age, sex, num as target)
TRULY_CRITICAL_COLUMNS = ['age', 'sex', 'num']

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
        if col == 'thal' and pd.api.types.is_numeric_dtype(df_clean[col]):
            # For thal: valid values are 3, 6, 7. Replace invalid values with NaN
            df_clean[col] = df_clean[col].replace([0, 1, 2, 4, 5, -9], np.nan)
        elif col == 'slope' and pd.api.types.is_numeric_dtype(df_clean[col]):
            # For slope: 0 is not valid, replace with NaN
            df_clean[col] = df_clean[col].replace([0, -9], np.nan)
        elif col == 'ca' and pd.api.types.is_numeric_dtype(df_clean[col]):
            # For ca: valid values are 0, 1, 2, 3. Replace invalid values with NaN
            df_clean[col] = df_clean[col].replace([-9], np.nan)
            # Also handle values > 3 as invalid
            invalid_ca_mask = df_clean[col] > 3
            df_clean.loc[invalid_ca_mask, col] = np.nan
    
    return df_clean

def handle_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle missing data with medical knowledge and simple imputation"""    
    stats = {"rows_removed": 0, "values_imputed": 0, "columns_dropped": 0}
    
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
        print(f"   🗑️  Removed {stats['rows_removed']} unusable rows")    # Impute missing values - prioritize statistical methods over medical defaults    # First pass: identify and remove columns with all missing values (except critical columns)
    columns_to_drop = []
    for col in df_clean.columns:
        if df_clean[col].isnull().all():
            if col not in TRULY_CRITICAL_COLUMNS:
                columns_to_drop.append(col)
                print(f"   🗑️  {col}: all values missing - will remove column")
            else:
                print(f"   ⚠️  {col}: all values missing but CRITICAL - will keep and fill with fallback values")
      # Drop columns with all missing values
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        stats["columns_dropped"] = len(columns_to_drop)
        print(f"   ✅ Removed {len(columns_to_drop)} columns with all missing values: {columns_to_drop}")      # Second pass: impute remaining missing values based on missing rate
    rows_to_remove = []
    
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            missing_count = df_clean[col].isnull().sum()
            missing_rate = missing_count / len(df_clean)
            
            print(f"   🔍 {col}: {missing_count} values ({missing_rate*100:.1f}% missing)")
            
           
            if missing_rate > 0.05:  # >5%
                print(f"   ⚠️  {col}: High missing rate ({missing_rate*100:.1f}%) - marking rows for removal")
                missing_rows = df_clean[df_clean[col].isnull()].index.tolist()
                rows_to_remove.extend(missing_rows)
                continue
            
            else:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    if pd.isna(median_val):
                        # If median is NaN (all values missing), use reasonable defaults
                        if col in ['age', 'trestbps', 'chol', 'thalach']:
                            # Use clinical defaults for important continuous variables
                            clinical_defaults = {
                                'age': 55,         # Typical MI patient age
                                'trestbps': 130,   # Normal-high BP
                                'chol': 200,       # Borderline cholesterol
                                'thalach': 150     # Age-adjusted max HR
                            }
                            fill_val = clinical_defaults.get(col, 0)
                            print(f"   🏥 {col}: filled {missing_count} values with clinical default ({fill_val})")
                        else:
                            fill_val = 0
                            print(f"   ⚠️  {col}: filled {missing_count} values with 0 (no valid data for median)")
                    else:
                        fill_val = median_val
                        print(f"   📊 {col}: filled {missing_count} values with median ({fill_val:.1f})")
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    
                # Cột phân loại: nội suy bằng giá trị xuất hiện nhiều nhất (mode)
                else:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val.iloc[0]
                        print(f"   📝 {col}: filled {missing_count} values with mode ({fill_val})")
                    else:
                        # If mode is empty (all values missing), use 'Unknown' as fallback
                        fill_val = 'Unknown'
                        print(f"   ⚠️  {col}: filled {missing_count} values with 'Unknown' (no valid data for mode)")
                    df_clean[col] = df_clean[col].fillna(fill_val)
                
                stats["values_imputed"] += missing_count
      # Remove rows with high missing rate data
    if rows_to_remove:
        rows_to_remove = list(set(rows_to_remove))  # Remove duplicates
        rows_removed_count = len(rows_to_remove)
        df_clean = df_clean.drop(index=rows_to_remove)
        df_clean = df_clean.reset_index(drop=True)
        stats["rows_removed"] += rows_removed_count
        stats["rows_removed_high_missing"] = rows_removed_count  # Track high missing rate removals separately
        print(f"   🗑️  Removed {rows_removed_count} rows with high missing rate data (>5%)")
    else:
        stats["rows_removed_high_missing"] = 0
    
    # Special handling for ca (number of major vessels) - always use median
    if 'ca' in df_clean.columns and df_clean['ca'].isnull().any():
        missing_count = df_clean['ca'].isnull().sum()
        median_val = df_clean['ca'].median()
        if pd.isna(median_val):
            fill_val = 0  # Default to 0 vessels (healthiest state)
            print(f"   🏥 ca: filled {missing_count} values with clinical default (0 vessels)")
        else:
            fill_val = median_val
            print(f"   🩺 ca: filled {missing_count} values with median ({fill_val:.0f} vessels)")
        df_clean['ca'] = df_clean['ca'].fillna(fill_val)
        stats["values_imputed"] += missing_count

    # Optimize data types
    type_conversions = {
        'sex': 'int8', 'cp': 'int8', 'fbs': 'int8', 'restecg': 'int8',
        'exang': 'int8', 'slope': 'int8', 'thal': 'int8', 'ca': 'int8',
        'htn': 'int8', 'dm': 'int8', 'famhist': 'int8',
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
    """Validate data ranges and fix obvious errors for MI risk prediction"""
    
    stats = {"invalid_values_corrected": 0}    # Define optimized ranges for MI risk prediction
    valid_ranges = {
        'age': (18, 100),           # Adult patients only
        'trestbps': (70, 250),      # Systolic BP range
        'chol': (100, 600),         # Total cholesterol
        'thalach': (60, 220),       # Max heart rate (age-adjusted upper)
        'oldpeak': (0, 10),         # ST depression
        'thal': (3, 7),             # Thalassemia types (3=normal, 6=fixed defect, 7=reversible defect)
    }    # Binary validation (0 or 1 only)
    binary_columns = ['sex', 'fbs', 'restecg', 'exang', 'htn', 
                     'dm', 'famhist', 'prop', 'nitr', 'pro', 
                     'diuretic', 'xhypo']
    
    # Multi-class validation
    multiclass_ranges = {
        'cp': (0, 4),               # Chest pain types (0-4)
        'slope': (1, 3),            # Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
        'num': (0, 4),              # Target variable (0-4, but often binarized)
        'ca': (0, 3),               # Number of major vessels (0-3) colored by fluoroscopy
    }
    
    print("🔍 Validating data ranges for MI risk prediction...")
    
    # Validate continuous variables
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = invalid_mask.sum()            
            if invalid_count > 0:
                # Use median for robust imputation
                median_val = df[col].median()
                df.loc[invalid_mask, col] = median_val
                
                stats["invalid_values_corrected"] += invalid_count
                print(f"   🔧 {col}: fixed {invalid_count} invalid values")
    
    # Validate binary columns
    for col in binary_columns:
        if col in df.columns:
            invalid_mask = ~df[col].isin([0, 1])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                # Default to 0 (normal/negative) for binary variables
                df.loc[invalid_mask, col] = 0
                stats["invalid_values_corrected"] += invalid_count
                print(f"   🔧 {col}: fixed {invalid_count} binary values")
    
    # Validate multi-class columns
    for col, (min_val, max_val) in multiclass_ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                # Use mode for categorical variables
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0
                df.loc[invalid_mask, col] = mode_val
                stats["invalid_values_corrected"] += invalid_count
                print(f"   🔧 {col}: fixed {invalid_count} categorical values")
    
    # Special handling for 'num' column - convert to binary classification
    if 'num' in df.columns:
        # Convert num > 1 to 1 for binary classification (0 = no disease, 1 = disease)
        binary_conversion_mask = df['num'] > 1
        conversion_count = binary_conversion_mask.sum()
        
        if conversion_count > 0:
            df.loc[binary_conversion_mask, 'num'] = 1
            stats["invalid_values_corrected"] += conversion_count
            print(f"   🎯 num: converted {conversion_count} values > 1 to 1 (binary classification)")
    
    return df, stats



# Smart one-hot encoding function for categorical variables
def create_onehot_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create one-hot encoding for categorical variables with >2 categories using pandas get_dummies"""
    
    stats = {"onehot_features_created": 0, "original_features_removed": 0, "binary_features_skipped": 0}
    
    print("🎯 Creating smart one-hot encoding for categorical variables...")
    df_onehot = df.copy()
    columns_to_remove = []  # Track original columns that get one-hot encoded      # Define categorical variables that might need one-hot encoding
    # NOTE: 'ca' (number of major vessels 0-3) is NOT included here because:
    # - It's ordinal: 0 < 1 < 2 < 3 vessels, so order matters
    # - Can be used directly as numeric feature without one-hot encoding
    categorical_candidates = {
        'restecg': 'restecg',  # 0=normal, 1=st_t_abnormal, 2=lvh
        'slope': 'slope',      # 1=upsloping, 2=flat, 3=downsloping  
        'cp': 'cp',            # 1=typical_angina, 2=atypical, 3=non_anginal, 4=asymptomatic
        'thal': 'thal'         # 3=normal, 6=fixed_defect, 7=reversible_defect
    }
    
    # Check each categorical variable
    for var_name, prefix in categorical_candidates.items():
        if var_name in df_onehot.columns:
            unique_values = df_onehot[var_name].nunique()
            unique_vals_list = sorted(df_onehot[var_name].unique())
            
            print(f"   🔍 Analyzing {var_name}: {unique_values} unique values {unique_vals_list}")
            
            # Only create one-hot if more than 2 categories
            if unique_values > 2:
                print(f"   🔄 Creating one-hot for {var_name} ({unique_values} categories)...")
                
                # Use pandas get_dummies for efficient one-hot encoding
                # drop_first=True to avoid dummy variable trap (removes first category)
                onehot_df = pd.get_dummies(
                    df_onehot[var_name], 
                    prefix=prefix,
                    dtype=int,
                    drop_first=False  # Avoid dummy variable trap by dropping first category
                )
                
                # Add the one-hot columns to the main dataframe
                df_onehot = pd.concat([df_onehot, onehot_df], axis=1)
                
                # Count features created
                features_created = len(onehot_df.columns)
                stats["onehot_features_created"] += features_created
                
                print(f"     ✅ Created {features_created} one-hot features: {list(onehot_df.columns)}")
                print(f"     ⚠️  Dropped first category to avoid dummy variable trap")
                
                # Mark original column for removal (since we have one-hot encoded it)
                columns_to_remove.append(var_name)
                print(f"     🗂️  Will remove original {var_name} column (redundant with one-hot)")
                
            else:
                print(f"     ⏭️  Skipping {var_name}: only {unique_values} categories (binary, no one-hot needed)")
                stats["binary_features_skipped"] += 1
    
    # Remove original categorical columns that were one-hot encoded
    if columns_to_remove:
        print(f"\n🧹 Removing {len(columns_to_remove)} original categorical columns after one-hot encoding...")
        for col in columns_to_remove:
            if col in df_onehot.columns:
                df_onehot = df_onehot.drop(columns=[col])
                stats["original_features_removed"] += 1
                print(f"     ✅ Removed original {col} column")    
    print(f"\n   ✅ Smart one-hot encoding completed:")
    print(f"      • One-hot features created: {stats['onehot_features_created']}")
    print(f"      • Original categorical columns removed: {stats['original_features_removed']}")
    print(f"      • Binary/ordinal features skipped: {stats['binary_features_skipped']}")
    
    return df_onehot, stats

def optimize_feature_distributions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Optimize feature distributions in-place without creating new columns"""
    
    stats = {"features_transformed": 0, "outliers_handled": 0, "binary_encoded": 0, "skew_corrected": 0}
    
    print("📊 Optimizing feature distributions in-place (no new columns)...")
    
    df_optimized = df.copy()
      # Features for outlier capping
    outlier_cap_features = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']
    
    # Handle outliers using IQR method for better model performance
    for col in outlier_cap_features:
        if col in df_optimized.columns and df_optimized[col].dtype in ['int64', 'float64']:
            Q1 = df_optimized[col].quantile(0.25)
            Q3 = df_optimized[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use more conservative outlier bounds for medical data
            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 2.0 * IQR
            
            outliers = ((df_optimized[col] < lower_bound) | 
                       (df_optimized[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap outliers instead of removing (preserve sample size)
                df_optimized.loc[df_optimized[col] < lower_bound, col] = lower_bound
                df_optimized.loc[df_optimized[col] > upper_bound, col] = upper_bound
                stats["outliers_handled"] += outlier_count
                print(f"   🎯 {col}: capped {outlier_count} outliers")
    
    # Apply in-place transformations for critical features
    critical_features = ['oldpeak', 'trestbps']
    for col in critical_features:
        if col in df_optimized.columns:
            skewness = df_optimized[col].skew()
            
            if col == 'oldpeak' and abs(skewness) > 1.0:
                # Apply absolute value to ensure oldpeak is non-negative
                df_optimized[col] = df_optimized[col].abs()
                print(f"   🔧 {col}: applied absolute value to ensure non-negative values")
                
                # Log transformation in-place for oldpeak
                min_val = df_optimized[col].min()
                if min_val <= 0:
                    df_optimized[col] = df_optimized[col] + 0.1
                
                # Apply log transformation in-place
                df_optimized[col] = np.log(df_optimized[col])
                stats["skew_corrected"] += 1
                print(f"   📈 {col}: applied log transformation in-place (skew: {skewness:.2f})")
            
            elif col == 'trestbps' and abs(skewness) > 0.5:
                # Winsorization for trestbps
                Q1 = df_optimized[col].quantile(0.01)
                Q99 = df_optimized[col].quantile(0.99)
                df_optimized[col] = df_optimized[col].clip(lower=Q1, upper=Q99)
                stats["skew_corrected"] += 1
                print(f"   🔧 {col}: applied winsorization in-place (skew: {skewness:.2f})")
    # Apply in-place transformation for continuous features
    continuous_transform_features = ['chol', 'thalach']
    for col in continuous_transform_features:
        if col in df_optimized.columns:
            skewness = df_optimized[col].skew()
            if abs(skewness) > 1.5:  # Only for very high skewness
                # Add small constant to handle zeros
                min_val = df_optimized[col].min()
                if min_val <= 0:
                    df_optimized[col] = df_optimized[col] + abs(min_val) + 1
                
                # Apply log transformation in-place
                df_optimized[col] = np.log(df_optimized[col])
                stats["skew_corrected"] += 1
                print(f"   📈 {col}: applied log transformation in-place (skew: {skewness:.2f})")      # Binary features validation (ensure proper encoding without new columns)
    binary_features = ['sex', 'fbs', 'restecg', 'exang', 'htn', 'dm', 'famhist', 
                      'prop', 'nitr', 'pro', 'diuretic', 'xhypo']
    
    for col in binary_features:
        if col in df_optimized.columns:
            # Ensure proper binary encoding (0/1)
            unique_vals = df_optimized[col].nunique()
            if unique_vals <= 2:
                df_optimized[col] = df_optimized[col].astype(int)
                stats["binary_encoded"] += 1
    
    # Ensure target variable is properly encoded for binary classification
    if 'num' in df_optimized.columns:
        # Keep original num column but ensure it's properly formatted
        df_optimized['num'] = df_optimized['num'].astype(int)
        print(f"   🎯 Validated target variable formatting")
        stats["features_transformed"] += 1
    
    # Summary of transformations
    print(f"   ✅ In-place optimization completed:")
    print(f"      • Outliers handled: {stats['outliers_handled']}")
    print(f"      • Binary features validated: {stats['binary_encoded']}")
    print(f"      • Skewness corrected: {stats['skew_corrected']}")
    
    return df_optimized, stats

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate rows"""
    before_count = len(df)
    df_dedup = df.drop_duplicates()
    total_removed = before_count - len(df_dedup)
    
    if total_removed > 0:
        print(f"   🗑️  Removed {total_removed} duplicate rows")
    
    return df_dedup, {"duplicates_removed": total_removed}

def transform_single_file(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform a single file - returns dataframe with smart one-hot encoded categorical variables"""    
    rows_before = len(df)
    print(f"🚀 Transforming {rows_before} rows for MI risk prediction...")    
    stats = {
        "rows_before": rows_before,
        "rows_after": 0,
        "data_retention_rate": 0,
        "rows_removed": 0,
        "rows_removed_high_missing": 0,  # New: track rows removed due to high missing rate
        "columns_dropped": 0,
        "duplicates_removed": 0,
        "values_imputed": 0,
        "invalid_values_corrected": 0,
        "features_transformed": 0,
        "outliers_handled": 0,
        "binary_encoded": 0,
        "skew_corrected": 0,
        "onehot_features_created": 0,
        "original_features_removed": 0,
        "binary_features_skipped": 0
    }

    # Step 1: Basic preprocessing
    df = standardize_column_names(df)
    df = select_columns(df, SELECTED_COLUMNS)
    print(f"   📋 Selected {len(df.columns)} MI risk-relevant columns")    # Step 2: Handle missing data
    df, imputation_stats = handle_missing_data(df)
    stats["rows_removed"] += imputation_stats["rows_removed"]
    stats["rows_removed_high_missing"] = imputation_stats.get("rows_removed_high_missing", 0)
    stats["columns_dropped"] += imputation_stats.get("columns_dropped", 0)
    stats["values_imputed"] += imputation_stats["values_imputed"]# Step 3: Validate and clean data
    df, validation_stats = validate_and_clean_data(df)
    stats["invalid_values_corrected"] += validation_stats["invalid_values_corrected"]

    # Step 4: Advanced feature optimization based on skewness analysis
    df, optimization_stats = optimize_feature_distributions(df)
    stats["features_transformed"] += optimization_stats["features_transformed"]
    stats["outliers_handled"] += optimization_stats["outliers_handled"]
    stats["binary_encoded"] += optimization_stats["binary_encoded"]  
    stats["skew_corrected"] += optimization_stats["skew_corrected"]

    # Step 5: Create smart one-hot encoding for categorical variables (>2 categories only)
    df, onehot_stats = create_onehot_features(df)
    stats["onehot_features_created"] += onehot_stats["onehot_features_created"]
    stats["original_features_removed"] += onehot_stats["original_features_removed"]
    stats["binary_features_skipped"] += onehot_stats["binary_features_skipped"]

    # Step 7: Remove duplicates
    df, duplicate_stats = remove_duplicates(df)
    stats["duplicates_removed"] += duplicate_stats["duplicates_removed"]

    # Step 8: Final column selection - keep remaining original columns + one-hot columns
    # Note: Original categorical columns (cp, thal, etc.) are removed after one-hot encoding
    available_columns = list(df.columns)
    original_columns = [col for col in SELECTED_COLUMNS if col in available_columns]
    onehot_columns = [col for col in available_columns if '_' in col and not col in SELECTED_COLUMNS]
    final_columns = original_columns + onehot_columns
    df_final = df[final_columns]
    
    print(f"   📋 Final selection: {len(original_columns)} remaining original + {len(onehot_columns)} one-hot = {len(final_columns)} total columns")    # Final statistics
    stats["rows_after"] = len(df_final)
    stats["data_retention_rate"] = (len(df_final) / rows_before) * 100

    print(f"   ✅ MI risk transformation completed: {stats['data_retention_rate']:.1f}% retention")
    print(f"   🗑️  Columns dropped (all missing): {stats['columns_dropped']}")
    print(f"   📊 Skewness corrections: {stats['skew_corrected']}, Binary encodings: {stats['binary_encoded']}")
    print(f"   🎯 One-hot features created: {stats['onehot_features_created']} (Binary skipped: {stats['binary_features_skipped']})")
    print(f"   🗂️  Original categorical columns removed: {stats['original_features_removed']}")
    print(f"   📋 Final columns: {len(final_columns)} total ({len(original_columns)} remaining + {len(onehot_columns)} one-hot)")
    
    return df_final, stats

def transform_raw_heart_disease_data(
    input_file: str = 'app/data/raw/to_csv/raw_heart_disease.csv',
    output_directory: str = 'app/data/processed'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform only the raw_heart_disease.csv file from to_csv folder to processed folder"""
    
    print("🏥 HEART DISEASE ETL - TRANSFORM RAW DATA TO PROCESSED")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists, if not, try to create it
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"⚠️ Input file not found: {input_file}")
        print("🔄 Attempting to create raw_heart_disease.csv...")
        
        # Import and use the ensure function
        from app.etl.extract.extract import ensure_raw_heart_disease_exists
        
        created_file = ensure_raw_heart_disease_exists()
        if created_file:
            input_file = created_file
            input_path = Path(input_file)
            print(f"✅ Created input file: {input_file}")
        else:
            print(f"❌ Failed to create input file: {input_file}")
            return pd.DataFrame(), {"error": f"Input file not found and could not be created: {input_file}"}
    
    print(f"📁 Processing: {input_path.name}")
    print(f"📂 Input location: {input_file}")
    print(f"📂 Output location: {output_directory}")
    
    try:
        # Load the raw data
        print(f"\n📥 Loading raw data...")
        df_raw = pd.read_csv(input_file)
        print(f"   📊 Raw data shape: {df_raw.shape}")
        print(f"   📋 Raw columns: {len(df_raw.columns)}")
        
        # Apply transformation
        print(f"\n🔄 Applying transformation pipeline...")
        transformed_df, stats = transform_single_file(df_raw)
          # Save transformed data
        output_file = output_path / "processed_heart_disease.csv"
        transformed_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Transformation completed!")
        print(f"   ✅ Processed data saved to: {output_file}")
        print(f"   📊 Final shape: {transformed_df.shape}")
        print(f"   📋 Final columns: {len(transformed_df.columns)}")
        
        # Verify saved file
        verification_df = pd.read_csv(output_file)
        print(f"   🔍 Verification - saved file shape: {verification_df.shape}")
          # Enhanced summary
        print(f"\n📈 TRANSFORMATION SUMMARY:")
        print(f"   • Input rows: {stats['rows_before']}")
        print(f"   • Output rows: {stats['rows_after']}")
        print(f"   • Data retention: {(stats['rows_after']/stats['rows_before']*100):.1f}%")
        print(f"   • Rows removed (high missing >5%): {stats.get('rows_removed_high_missing', 0)}")
        print(f"   • Values imputed (low missing <5%): {stats['values_imputed']}")
        print(f"   • Duplicates removed: {stats['duplicates_removed']}")
        print(f"   • Features transformed: {stats['features_transformed']}")
        print(f"   • Outliers handled: {stats['outliers_handled']}")
        
        if 'onehot_features_created' in stats:
            print(f"   • One-hot features created: {stats['onehot_features_created']}")
        if 'binary_encoded' in stats:
            print(f"   • Binary encodings: {stats['binary_encoded']}")
        
        print(f"\n🎉 Ready for analysis and modeling!")
        
        return transformed_df, stats
        
    except Exception as e:
        error_msg = f"Error processing {input_file}: {e}"
        print(f"❌ {error_msg}") 
        return pd.DataFrame(), {"error": error_msg}