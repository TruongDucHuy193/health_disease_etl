import os
import requests
import pandas as pd
import numpy as np

def extract_heart_disease_data():
    '''URL of the UCI heart disease dataset repository'''
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    
    # List of data files to download
    data_files = [
        "new.data",
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
        
def convert_to_heart_disease_csv():
    """Convert new.data file to raw_heart_disease.csv with proper labeling"""
    raw_directory = os.path.join("app", "data", "raw")
    output_dir = os.path.join("app", "data", "raw", "to_csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define column names for the heart disease dataset (76 attributes)
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
    ]    # Process the new.data file
    raw_file_path = os.path.join(raw_directory, "new.data")
    output_file = os.path.join(output_dir, "raw_heart_disease.csv")
    
    if not os.path.exists(raw_file_path):
        print(f"❌ File not found: {raw_file_path}")
        return None
        
    print(f"Processing new.data...")
    
    # Read the file with different encodings
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
      # Save directly as raw_heart_disease.csv
    df.to_csv(output_file, index=False, escapechar='\\')
    print(f"✅ Created raw_heart_disease.csv: {output_file}")
    print(f"   📊 Total rows: {len(df)}")
    
    return df


def extract_all_data():
    """Complete extraction pipeline: download -> convert to CSV with labels"""
    
    print("🚀 Starting Heart Disease Data Extraction Pipeline...")
    
    # Step 1: Download raw data files
    print("\n📥 Step 1: Downloading raw data files...")
    extract_heart_disease_data()
    
    # Step 2: Convert to labeled CSV format
    print("\n🔄 Step 2: Converting to labeled CSV format...")
    heart_disease_df = convert_to_heart_disease_csv()
    
    print("\n🎉 Extraction pipeline completed successfully!")
    return heart_disease_df

def ensure_raw_heart_disease_exists():
    """Ensure raw_heart_disease.csv exists, create if missing"""
    
    raw_file_path = os.path.join("app", "data", "raw", "to_csv", "raw_heart_disease.csv")
    
    # Check if file exists
    if os.path.exists(raw_file_path):
        print(f"✅ raw_heart_disease.csv already exists: {raw_file_path}")
        return raw_file_path
    
    print(f"⚠️ raw_heart_disease.csv not found, creating it...")
    
    # Check if the source new.data file exists
    new_data_path = os.path.join("app", "data", "raw", "new.data")
    if not os.path.exists(new_data_path):
        print("📥 new.data file not found, running full extraction pipeline...")
        heart_disease_df = extract_all_data()
    else:
        print("🔗 new.data file found, converting to raw_heart_disease.csv...")
        heart_disease_df = convert_to_heart_disease_csv()
    
    if heart_disease_df is not None:
        print(f"✅ raw_heart_disease.csv created successfully: {raw_file_path}")
        return raw_file_path
    else:
        print("❌ Failed to create raw_heart_disease.csv")
        return None
