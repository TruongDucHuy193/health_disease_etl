import os
import requests
import pandas as pd
import numpy as np

def extract_heart_disease_data():
    '''URL of the UCI heart disease dataset repository'''
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    
    # List of data files to download
    data_files = [
        "hungarian.data",
        "switzerland.data",
        "long-beach-va.data",
    ]

   # Create the directory to store raw data files
    raw_directory = os.path.join("app", "data", "raw")
    os.makedirs(raw_directory, exist_ok=True)

    # Download file
    for filename in data_files:
        data_url = base_url + filename
        local_data_path = os.path.join(raw_directory, filename)
        
        try:
            # Attempt to download file with a timeout
            r = requests.get(data_url, timeout=30)
            r.raise_for_status()  # Raise exception for HTTP errors
        except Exception as e:
            print(f"⚠️ Can't download {filename}: {e}")
            continue
        
        # Save file
        with open(local_data_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Download complete: {filename}")
        
def extract_data_to_csv():
    raw_directory = os.path.join("app", "data", "raw")
    output_dir = os.path.join("app", "data", "raw", "to_csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define column names for the heart disease dataset (76 attributes)'''
    col_names = [
        "id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden",
        "cp", "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs", "dm",
        "famhist", "restecg", "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr",
        "pro", "diuretic", "proto", "thaldur", "thaltime", "met", "thalach",
        "thalrest", "tpeakbps", "tpeakbpd", "dummy", "trestbpd", "exang", "xhypo",
        "oldpeak", "slope", "rldv5", "rldv5e", "ca", "restckm", "exerckm",
        "restef", "restwm", "exeref", "exerwm", "thal", "thalsev", "thalpul",
        "earlobe", "cmo", "cday", "cyr", "num", "lmt", "ladprox", "laddist",
        "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1",
        "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name"
    ]

    # Process each .data file in the raw directory
    for filename in os.listdir(raw_directory):
        if filename.endswith(".data"):
            raw_file_path = os.path.join(raw_directory, filename)
            output_file = os.path.join(output_dir, filename.replace('.data', '.csv'))   
            print(f"Processing {filename}...")
            
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    with open(raw_file_path, 'r', encoding=encoding) as f:
                        raw_data = f.read()
                    print(f"  Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    print(f"  Failed to read with {encoding} encoding, trying next...")
            else:
                # Fallback to binary mode with error replacement if all encodings fail
                with open(raw_file_path, 'rb') as f:
                    raw_data = f.read().decode('latin-1', errors='replace')
                print("  Used binary mode with latin-1 fallback")
            
            # Parse the raw data format into structured lines
            lines = []
            current_line = []
            
            # The raw data format can have values spread across multiple lines
            for line in raw_data.strip().split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                    
                if "name" in line: 
                    current_line.append("name")
                    lines.append(' '.join(current_line))  
                    current_line = []  
                else:
                    values = line.split()
                    current_line.extend(values)
            
            # Add the last record if it exists
            if current_line:
                lines.append(' '.join(current_line))
            
            # Convert parsed lines to rows with fixed number of columns
            data_rows = []
            for line in lines:
                values = line.split()
                
                # Ensure each row has exactly 76 columns
                if len(values) < 76:
                    # Pad with NaN if there are fewer than 76 values
                    values.extend([np.nan] * (76 - len(values)))
                elif len(values) > 76:
                    # Truncate if there are more than 76 values
                    values = values[:76]  
                    
                data_rows.append(values)
            
            # Create a DataFrame with the proper column names
            df = pd.DataFrame(data_rows, columns=col_names)
            
            # Save to CSV
            df.to_csv(output_file, index=False, escapechar='\\')
            print(f"✅ Created CSV: {output_file}")

def merge_csv_files():
    """Merge all individual CSV files into one combined heart_disease_raw.csv file"""
    
    csv_dir = os.path.join("app", "data", "raw", "to_csv")
    output_dir = os.path.join("app", "data", "raw","to_csv")
    output_file = os.path.join(output_dir, "heart_disease_raw.csv")
    
    print("\n🔄 Merging CSV files...")
    
    # Check if the merged file already exists
    if os.path.exists(output_file):
        print(f"✅ heart_disease_raw.csv already exists at: {output_file}")
        
        # Load existing file and return it
        try:
            existing_df = pd.read_csv(output_file)
            print(f"   📊 Existing file has {len(existing_df)} rows")
            return existing_df
        except Exception as e:
            print(f"   ⚠️ Warning: Could not read existing file ({e}), will recreate...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store all dataframes
    all_dataframes = []
    
    # Process each CSV file in the to_csv directory (exclude the target file)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and f != 'raw_heart_disease.csv']
    
    if not csv_files:
        print("❌ No individual CSV files found to merge!")
        print("💡 Run extract_data_to_csv() first to create individual CSV files")
        return None
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Add source column to track which dataset each row came from
            dataset_name = csv_file.replace('.csv', '')
            df['source_dataset'] = dataset_name
            
            all_dataframes.append(df)
            print(f"   ✅ Loaded {csv_file}: {len(df)} rows")
            
        except Exception as e:
            print(f"   ❌ Error reading {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("❌ No valid CSV files to merge!")
        return
    
    # Combine all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
      # Save merged dataframe
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully merged {len(all_dataframes)} files into heart_disease_raw.csv")
    print(f"   📊 Total rows: {len(merged_df)}")
    print(f"   📁 Output file: {output_file}")
    
    # Show summary by dataset
    print(f"\n📋 Dataset Summary:")
    dataset_counts = merged_df['source_dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        print(f"   • {dataset}: {count} rows")
    
    return merged_df

def extract_all_data():
    """Complete extraction pipeline: download -> convert to CSV -> merge"""
    
    print("🚀 Starting Heart Disease Data Extraction Pipeline...")
    
    # Step 1: Download raw data files
    print("\n📥 Step 1: Downloading raw data files...")
    extract_heart_disease_data()
    
    # Step 2: Convert to CSV format
    print("\n🔄 Step 2: Converting to CSV format...")
    extract_data_to_csv()
    
    # Step 3: Merge all CSV files
    print("\n🔗 Step 3: Merging CSV files...")
    merged_df = merge_csv_files()
    
    print("\n🎉 Extraction pipeline completed successfully!")
    return merged_df

def ensure_raw_heart_disease_exists():
    """Ensure heart_disease_raw.csv exists, create if missing"""
    
    raw_file_path = os.path.join("app", "data", "raw", "to_csv", "heart_disease_raw.csv")
    
    # Check if file exists
    if os.path.exists(raw_file_path):
        print(f"✅ heart_disease_raw.csv already exists: {raw_file_path}")
        return raw_file_path
    
    print(f"⚠️ heart_disease_raw.csv not found, creating it...")
    
    # Check if individual CSV files exist
    csv_dir = os.path.join("app", "data", "raw", "to_csv")
    if not os.path.exists(csv_dir):
        print(f"📁 Creating directory: {csv_dir}")
        os.makedirs(csv_dir, exist_ok=True)
    
    individual_csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and f != 'heart_disease_raw.csv']
    
    if not individual_csvs:
        print("📥 No individual CSV files found, running full extraction pipeline...")
        merged_df = extract_all_data()
    else:
        print("🔗 Individual CSV files found, merging them...")
        merged_df = merge_csv_files()
    
    if merged_df is not None:
        print(f"✅ heart_disease_raw.csv created successfully: {raw_file_path}")
        return raw_file_path
    else:
        print("❌ Failed to create heart_disease_raw.csv")
        return None
