# Heart Disease ETL Pipeline

This project implements an ETL (Extract, Transform, Load) pipeline for the UCI Heart Disease dataset. The pipeline extracts raw data from the UCI Machine Learning Repository, transforms it through comprehensive cleaning and feature engineering for machine learning applications, and loads it into CSV format for analysis. The system ensures high data quality through medical knowledge-based validation, intelligent missing value handling, and detailed statistical reporting of the entire ETL process.

## Design Choices

This ETL pipeline was designed with the following principles in mind:

**Modularity**: The code is organized into separate modules for extraction, transformation, and loading, making it easy to maintain and extend with clear separation of concerns.

**Medical Domain Knowledge**: The pipeline incorporates medical expertise in data validation, feature selection, and missing value imputation strategies specific to cardiovascular risk assessment.

**Data Quality Assurance**: Thorough validation and cleaning processes ensure the output data meets quality standards for machine learning with detailed statistical reports of transformations applied.

**Machine Learning Ready**: The transformation process includes intelligent feature engineering, one-hot encoding for categorical variables, and optimization for MI (Myocardial Infarction) risk prediction models.

**Observability**: Comprehensive logging provides visibility into each step of the ETL process, recording data quality metrics, transformation statistics, and any data issues encountered.

**Flexibility**: Configurable column selection and transformation parameters allow customization for different analysis requirements.

## ETL Process Details

### Extract

- **Data Source**: Heart Disease dataset from the UCI Machine Learning Repository
- **Dataset Information**: While UCI provides 4 heart disease databases (Cleveland, Hungarian, Switzerland, Long Beach VA), only the `new.data` file contains the complete combined dataset that can be properly processed
- **Method**: Downloads raw data files directly from UCI repository using HTTP requests
- **File Processing**: Converts the raw `.data` format to structured CSV with proper column labeling (76 attributes)
- **Data Parsing**: Handles multi-line records and ensures exactly 76 columns per row
- **Column Naming**: Maps raw data to meaningful column names including medical terminology
- **Error Handling**: Gracefully handles connection timeouts, HTTP errors, and various text encodings (utf-8, latin-1, iso-8859-1)
- **Data Validation**: Ensures proper file structure with 76 attributes as per UCI specification
- **Record Processing**: Handles records that span multiple lines and normalizes to single-line format
- **Logging**: Records download status, file sizes, and conversion statistics

### Transform

**Data Cleaning**:
- Standardizes column names to lowercase format for consistency
- Handles multiple missing value indicators (`-9`, `NULL`, `?`, `NaN`, etc.)
- Implements medical knowledge-based missing value imputation
- Validates data ranges based on medical standards (e.g., age: 18-100, blood pressure: 70-250 mmHg)
- Removes duplicate records and empty rows
- Converts target variable to binary classification (0: no disease, 1: disease)

**Feature Engineering**:
- Selects 22 most relevant features for MI risk prediction from original 76 attributes
- Creates smart one-hot encoding only for categorical variables with >2 categories
- Preserves ordinal variables (ca: 0-3 vessels) without one-hot encoding
- Applies IQR-based outlier capping for continuous variables (age, chol, trestbps, thalach, oldpeak)
- Optimizes data types to int8 for binary/categorical variables for memory efficiency
- Removes original categorical columns after one-hot encoding

**Advanced Processing**:
- Handles ordinal vs nominal categorical variables appropriately
- Smart one-hot encoding for: cp (chest pain), thal (thalassemia), restecg (resting ECG), slope
- Preserves ordinal nature of ca (number of major vessels: 0-3)
- Validates binary encodings for clinical variables (sex, fbs, exang, htn, dm, famhist, etc.)
- Final column selection includes original + one-hot encoded features

### Handling Missing and Inconsistent Data

Our approach to handling missing and inconsistent data follows medical best practices:

**Missing Values Strategy**:
- **Low Missing Rate (<5%)**:
  - Numeric columns: Impute with median values (robust to outliers)
  - Special handling for 'ca' (number of major vessels): Always use median imputation
  - Binary/categorical columns: Use appropriate statistical imputation
- **High Missing Rate (>5%)**:
  - Remove rows with excessive missing data to maintain data integrity
  - Remove columns with all missing values (except critical columns)
  - Log all removals with detailed statistics for transparency

**Medical Data Validation**:
- **Physiological Ranges**: Age (18-100), Blood Pressure (70-250 mmHg), Cholesterol (100-600 mg/dl)
- **Heart Rate**: Maximum heart rate (60-220 bpm), ST depression (0-10)
- **Clinical Variables**: Validates chest pain types (0-4), ECG results, slope types (1-3)
- **Thalassemia**: Validates thalassemia types (3=normal, 6=fixed defect, 7=reversible defect)
- **Binary Encodings**: Ensures proper 0/1 encoding for medical condition indicators
- **Ordinal Variables**: Number of major vessels (0-3) preserved as ordinal
- **Target Variable**: Converts multi-class heart disease severity (0-4) to binary classification

**Data Quality Assurance**:
- Comprehensive range validation with medical knowledge
- IQR-based outlier detection and capping for continuous variables
- Duplicate record identification and removal
- Data type optimization (int8 for binary/categorical variables)
- Missing value indicator cleaning ('-9', 'NULL', '?', 'NaN', etc.)
- Transformation statistics and quality metrics reporting

### Load

- **Output Format**: Processed CSV files optimized for analysis and machine learning
- **Directory Structure**: Organized data storage with raw, processed, and demo folders
- **Data Verification**: Post-processing validation to ensure data integrity
- **Metadata**: Includes transformation statistics and data lineage information
- **Analysis Ready**: Output formatted for immediate use in Jupyter notebooks and ML pipelines

## Technical Requirements

- **Language**: Python 3.10+
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, scipy, requests
- **Analysis Tools**: Jupyter notebooks for exploratory data analysis
- **Machine Learning**: scikit-learn for advanced analysis (optional)
- **Data Processing**: Vectorized operations for efficient large dataset handling
- **File Management**: Automated directory creation and file organization

## Features

**Automated ETL Pipeline**: End-to-end process from raw data download to analysis-ready datasets
**Medical Domain Intelligence**: Incorporates cardiovascular medicine knowledge in data processing
**Statistical Reporting**: Comprehensive data quality metrics and transformation statistics
**Machine Learning Optimization**: Features engineered specifically for heart disease prediction
**Interactive Analysis**: Jupyter notebooks with detailed exploratory data analysis
**Robust Error Handling**: Comprehensive exception handling with informative error messages
**Data Lineage Tracking**: Complete audit trail of all transformations applied

## Setup and Usage

### Prerequisites
- Python 3.10 

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd heart_disease_etl
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

   Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scipy requests jupyter scikit-learn
```

### Running the Pipeline

1. **Execute the complete ETL pipeline**:
```bash
python main.py
```

2. **Or run individual components**:
```python
# Extract data
from app.etl.extract.extract import extract_all_data
extract_all_data()

# Transform data
from app.etl.transform.transform import transform_raw_heart_disease_data
transformed_df, stats = transform_raw_heart_disease_data()
```

### Checking the Results

**Verify processed data**:
```bash
# Check processed data file
ls -la app/data/processed/processed_heart_disease.csv

# View first few rows
head app/data/processed/processed_heart_disease.csv
```

**Run analysis notebooks**:
```bash
jupyter notebook app/notebooks/
```

## Configuration Options

You can customize the pipeline behavior by modifying these parameters in the transformation module:

- **SELECTED_COLUMNS**: List of features to include in the final dataset
- **TRULY_CRITICAL_COLUMNS**: Columns that must have values (age, sex, target)
- **Validation Ranges**: Medical ranges for data validation
- **Missing Value Thresholds**: Percentage thresholds for missing data handling
- **One-Hot Encoding**: Categorical variables to be one-hot encoded

## Data Schema

### Input Data (76 attributes)
Complete UCI heart disease dataset including demographics, clinical measurements, medical history, and test results.

**Note**: The UCI Heart Disease Database contains 4 separate databases:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology  
- V.A. Medical Center (Long Beach)
- University Hospital (Zurich, Switzerland)

However, this ETL pipeline processes only the `new.data` file, which contains the complete combined dataset with all 76 attributes properly formatted for processing. The individual database files have incomplete or inconsistent formatting that makes them unsuitable for automated processing.

### Output Data (Selected Features)
22 most relevant features for MI risk prediction:
```python
['age', 'sex', 'num', 'cp', 'thal', 'ca', 'oldpeak', 'exang', 
 'trestbps', 'chol', 'thalach', 'slope', 'restecg', 'htn', 'dm', 
 'famhist', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'xhypo']
```

## Analysis Notebooks

- **1_rlt_heath_factors.ipynb**: Relationship analysis between health factors and heart disease
- **analyse.ipynb**: Comprehensive exploratory data analysis
- **review_data.ipynb**: Data quality assessment and validation

## Data Quality Metrics

The pipeline provides detailed reporting on:
- Data retention rates after cleaning
- Missing value imputation statistics
- Outlier detection and handling counts
- Feature engineering transformation logs
- Binary encoding validation results
- One-hot encoding creation statistics

## Project Structure

```
heart_disease_etl/
├── main.py                     # Main execution script
├── README.md                  
├── app/
│   ├── data/
│   │   ├── processed/          # Transformed
│   │   └── raw/                # Raw data for observation
│   ├── etl/
│   │   ├── extract/            # Data extraction module
│   │   ├── transform/          # Data transformation module
│   │   └── load/               # Data loading utilities
│   └── notebooks/              # Jupyter notebooks for analysis
└── SQL/                        # SQL scripts and database 
```

## References
- UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- Cleveland Clinic Foundation, Hungarian Institute of Cardiology, V.A. Medical Center, University Hospital Zurich
