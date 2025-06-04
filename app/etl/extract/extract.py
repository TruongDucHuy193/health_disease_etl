import os
import requests
import pandas as pd
import numpy as np

def extract_heart_disease_data():
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    data_files = [
        "cleveland.data",
        "hungarian.data",
        "switzerland.data",
    ]

    raw_dir = os.path.join("app", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # crawl .data
    for filename in data_files:
        data_url = base_url + filename
        local_data_path = os.path.join(raw_dir, filename)
        
        try:
            r = requests.get(data_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"⚠️ Can't download {filename}: {e}")
            continue
        
        with open(local_data_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Download complete: {filename}")
        
def extract_data_to_csv():
    raw_dir = os.path.join("app", "data", "raw")
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
    ]

    for filename in os.listdir(raw_dir):
        if filename.endswith(".data"):
            raw_file_path = os.path.join(raw_dir, filename)
            output_file = os.path.join(output_dir, filename.replace('.data', '.csv'))
            
            print(f"Processing {filename}...")
            
            # Try different encodings to read the file
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    with open(raw_file_path, 'r', encoding=encoding) as f:
                        raw_data = f.read()
                    print(f"  Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    print(f"  Failed to read with {encoding} encoding, trying next...")
            else:
                with open(raw_file_path, 'rb') as f:
                    raw_data = f.read().decode('latin-1', errors='replace')
                print("  Used binary mode with latin-1 fallback")
            
            lines = []
            current_line = []
            
            for line in raw_data.strip().split('\n'):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                if "name" in line:
                    current_line.append("name")
                    lines.append(' '.join(current_line))
                    current_line = []
                else:
                    values = line.split()
                    current_line.extend(values)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            data_rows = []
            for line in lines:
                values = line.split()
                
                if len(values) < 76:
                    values.extend([np.nan] * (76 - len(values)))
                elif len(values) > 76:
                    values = values[:76]  # Truncate if too many
                    
                data_rows.append(values)
            
            df = pd.DataFrame(data_rows, columns=col_names)
              
            # Save to CSV
            df.to_csv(output_file, index=False, escapechar='\\')
            print(f"✅ Created CSV: {output_file}")