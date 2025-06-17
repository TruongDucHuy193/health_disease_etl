import pandas as pd
from pathlib import Path
import os
import sys
from typing import Dict, Any

project_root = Path(__file__).parents[3]  
sys.path.insert(0, str(project_root))

# Now the app module can be found
# Removed import of transform_heart_disease_data as it's no longer needed

def load_to_csv(transformed_data, output_directory='app/data/processed'):
    """Save transformed DataFrame to single file: processed_heart_disease.csv"""
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the single output file
    output_file = output_path / "processed_heart_disease.csv"
    
    # If transformed_data is a DataFrame (from transform_raw_heart_disease_data)
    if isinstance(transformed_data, pd.DataFrame):
        transformed_data.to_csv(output_file, index=False)
        print(f"✅ Saved processed_heart_disease.csv with {len(transformed_data)} rows")
        print(f"📁 File location: {output_file}")
        return output_file
      # If transformed_data is a dictionary of DataFrames, take the main one
    elif isinstance(transformed_data, dict):
        main_df = None
        
        # Priority 1: Look for processed file
        if any('heart_disease_processed' in key for key in transformed_data.keys()):
            key = next(key for key in transformed_data.keys() if 'heart_disease_processed' in key)
            main_df = transformed_data[key]
            print(f"📋 Using processed data: {key}")
        
        # Priority 2: Take the first DataFrame
        else:
            first_key = list(transformed_data.keys())[0]
            main_df = transformed_data[first_key]
            print(f"📋 Using first available data: {first_key}")
        
        # Save the selected DataFrame
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

def merge_processed_files(
    processed_directory: str = 'app/data/processed',
    output_file: str = 'app/data/processed/processed_heart_disease.csv'
) -> Dict[str, Any]:
    """Merge all processed CSV files into a single processed_heart_disease.csv file."""
    # Create the output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all processed CSV files (exclude the target file itself)
    input_path = Path(processed_directory)
    processed_files = list(input_path.glob('processed_*.csv'))
    
    # Remove the target file from the list if it exists
    target_file = Path(output_file)
    processed_files = [f for f in processed_files if f != target_file]
    
    if not processed_files:
        return {"error": "No processed CSV files found to merge"}
    
    stats = {
        "files_merged": 0,
        "total_rows": 0,
        "rows_after_dedup": 0,
        "duplicates_removed": 0
    }
    
    all_dfs = []
    
    # Read processed files
    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            
            stats["files_merged"] += 1
            stats["total_rows"] += len(df)
            all_dfs.append(df)
            print(f"📄 Read {len(df)} rows from {file_path.name}")
            
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            continue
    
    if not all_dfs:
        return {"error": "None of the files could be read"}
    
    # Concatenate all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    duplicates_removed = before_dedup - len(merged_df)
    
    stats["rows_after_dedup"] = len(merged_df)
    stats["duplicates_removed"] = duplicates_removed
    
    # Save the merged file as processed_heart_disease.csv
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Saved processed_heart_disease.csv with {len(merged_df)} rows")
    print(f"📁 File location: {output_file}")
    
    return stats

def load_processed_heart_disease_data(
    input_file: str = 'app/data/raw/to_csv/heart_disease_raw.csv',
    output_directory: str = 'app/data/processed'
) -> str:
    """Complete pipeline: Transform raw data and save to processed_heart_disease.csv"""
    
    print("🚀 COMPLETE ETL PIPELINE: RAW → PROCESSED")
    print("=" * 50)
    
    # Import the transform function
    from app.etl.transform.transform import transform_raw_heart_disease_data
    
    # Step 1: Transform the raw data
    print("🔄 Step 1: Transforming raw data...")
    transformed_df, stats = transform_raw_heart_disease_data(input_file, output_directory)
    
    if transformed_df.empty:
        print("❌ Transformation failed - no data to save")
        return None
    
    # Step 2: Save to standard filename
    print("\n💾 Step 2: Saving to processed_heart_disease.csv...")
    output_file = load_to_csv(transformed_df, output_directory)
    
    # Step 3: Summary
    print(f"\n✅ ETL PIPELINE COMPLETED!")
    print(f"   📊 Processed {stats['rows_before']} → {stats['rows_after']} rows")
    print(f"   📁 Output file: {output_file}")
    print(f"   🎯 Data retention: {stats['data_retention_rate']:.1f}%")
    
    return str(output_file) if output_file else None

def load_existing_processed_data(file_path: str = 'app/data/processed/processed_heart_disease.csv') -> pd.DataFrame:
    """Load the processed heart disease data for analysis"""
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        print("💡 Run load_processed_heart_disease_data() first to create the processed data")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded processed heart disease data: {df.shape}")
        print(f"📋 Columns: {len(df.columns)}")
        print(f"📊 Sample columns: {list(df.columns[:10])}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Run the complete ETL pipeline
    print("🏥 HEART DISEASE ETL - LOAD TO CSV")
    print("=" * 40)
    
    # Option 1: Run complete pipeline (transform + save)
    output_file = load_processed_heart_disease_data()
    
    if output_file:
        print(f"\n🎉 Success! Processed data saved to: {output_file}")
        
        # Option 2: Load and display sample of processed data
        print(f"\n📊 Loading processed data for verification...")
        df = load_existing_processed_data()
        
        if not df.empty:
            print(f"\n📋 DATA OVERVIEW:")
            print(f"   • Shape: {df.shape}")
            print(f"   • Columns: {list(df.columns)}")
            print(f"   • Target distribution: {df['num'].value_counts().to_dict()}")
            print(f"\n📝 First 5 rows:")
            print(df.head())
    else:
        print("❌ ETL Pipeline failed")
