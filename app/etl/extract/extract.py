import os
import requests
import pandas as pd
import numpy as np

def extract_heart_disease_data():
    '''URL of the UCI heart disease dataset repository'''
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    
    # List of data files to download
    data_files = [
        "cleveland.data",
        "hungarian.data",
        "switzerland.data",
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