import pandas as pd
from pathlib import Path
import os
import sys
from typing import Dict, Any

# Setup project path
project_root = Path(__file__).parents[3]  
sys.path.insert(0, str(project_root))

def load_to_csv(transformed_data, output_directory='app/data/processed'):
    """Save transformed DataFrame to processed_heart_disease.csv"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the output file
    output_file = output_path / "processed_heart_disease.csv"
    
    # Handle DataFrame input
    if isinstance(transformed_data, pd.DataFrame):
        transformed_data.to_csv(output_file, index=False)
        print(f"✅ Saved processed_heart_disease.csv with {len(transformed_data)} rows")
        print(f"📁 File location: {output_file}")
        return output_file
    
    # Handle dictionary input (legacy support)
    elif isinstance(transformed_data, dict):
        # Find the main DataFrame
        main_df = None
        
        if any('processed' in key.lower() for key in transformed_data.keys()):
            key = next(key for key in transformed_data.keys() if 'processed' in key.lower())
            main_df = transformed_data[key]
            print(f"📋 Using processed data: {key}")
        else:
            first_key = list(transformed_data.keys())[0]
            main_df = transformed_data[first_key]
            print(f"📋 Using first available data: {first_key}")
        
        if main_df is not None:
            main_df.to_csv(output_file, index=False)
            print(f"✅ Saved processed_heart_disease.csv with {len(main_df)} rows")
            print(f"📁 File location: {output_file}")
            return output_file
        else:
            print("❌ No valid DataFrame found in dictionary")
            return None
    
    else:
        print("❌ Error: transformed_data must be a DataFrame or dictionary of DataFrames")
        return None

def load_processed_heart_disease_data(
    input_file: str = 'app/data/raw/to_csv/raw_heart_disease.csv',
    output_directory: str = 'app/data/processed'
) -> str:
    """Complete ETL pipeline: Transform raw_heart_disease.csv → processed_heart_disease.csv"""
    
    print("🚀 COMPLETE ETL PIPELINE: RAW → PROCESSED")
    print("=" * 50)
    
    # Import transform function
    from app.etl.transform.transform import transform_raw_heart_disease_data
    
    # Step 1: Transform raw data
    print("🔄 Step 1: Transforming raw data...")
    transformed_df, stats = transform_raw_heart_disease_data(input_file, output_directory)
    
    if transformed_df.empty:
        print("❌ Transformation failed - no data to save")
        return None
    
    # Step 2: Save to processed file
    print("\n💾 Step 2: Saving to processed_heart_disease.csv...")
    output_file = load_to_csv(transformed_df, output_directory)
    
    # Step 3: Summary
    print(f"\n✅ ETL PIPELINE COMPLETED!")
    print(f"   📊 Processed {stats['rows_before']} → {stats['rows_after']} rows")
    print(f"   📁 Output: {output_file}")
    print(f"   🎯 Retention: {stats['data_retention_rate']:.1f}%")
    
    return str(output_file) if output_file else None

def load_existing_processed_data(file_path: str = 'app/data/processed/processed_heart_disease.csv') -> pd.DataFrame:
    """Load existing processed heart disease data for analysis"""
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        print("💡 Run load_processed_heart_disease_data() first to create the processed data")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded processed data: {df.shape}")
        print(f"📋 Columns: {len(df.columns)}")
        print(f"📊 Sample columns: {list(df.columns[:10])}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

