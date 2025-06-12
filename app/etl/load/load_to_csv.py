import pandas as pd
from pathlib import Path
import os
import sys
from typing import Dict, Any

project_root = Path(__file__).parents[3]  
sys.path.insert(0, str(project_root))

# Now the app module can be found
from app.etl.transform.transform import transform_heart_disease_data

def load_to_csv(transformed_data, output_directory='app/data/processed'):
    """Save transformed DataFrames to CSV files and merge regional datasets"""
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each DataFrame to a CSV file
    for filename, df in transformed_data.items():
        file_path = output_path / filename
        df.to_csv(file_path, index=False)
        print(f"Saved {filename} with {len(df)} rows")
    
    # Find the regional datasets (assuming they have 'region' in the name)
    region_files = [f for f in transformed_data.keys() if 'region' in f.lower()]
    
    # Merge the 3 regional datasets
    if len(region_files) == 3:
        print(f"\nMerging 3 regional datasets:")
        for rf in region_files:
            print(f"- {rf}")
        
        # Combine the regional DataFrames
        region_dfs = [transformed_data[rf] for rf in region_files]
        merged_df = pd.concat(region_dfs, ignore_index=True)
        
        # Save the merged dataset
        merged_path = output_path / "heart_disease_data.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"\nCreated heart_disease_data.csv with {len(merged_df)} rows")
    else:
        print(f"\nWarning: Expected 3 regional datasets, but found {len(region_files)}")
    
    print(f"\nCompleted: All files saved to {output_path}")

def merge_processed_files(
    processed_directory: str = 'app/data/processed',
    output_file: str = 'app/data/processed/heart_disease_data.csv'
) -> Dict[str, Any]:
    """Merge all processed CSV files into a single file and generate summary statistics."""
    # Create the output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all processed CSV files
    input_path = Path(processed_directory)
    processed_files = list(input_path.glob('processed_*.csv'))
    
    if not processed_files:
        return {"error": "No processed CSV files found"}
    
    stats = {
        "files_merged": 0,
        "total_rows": 0,
        "rows_after_dedup": 0,
        "duplicates_removed": 0
    }
    
    all_dfs = []
    
    # Read processed file
    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            
            stats["files_merged"] += 1
            stats["total_rows"] += len(df)
            all_dfs.append(df)
            print(f"Read {len(df)} rows from {file_path.name}")
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
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
    
    # Save the merged file
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged file with {len(merged_df)} rows to {output_file}")
    
    return stats
