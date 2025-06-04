from app.etl.extract.extract import extract_heart_disease_data, extract_data_to_csv
from app.etl.transform.transform import transform_heart_disease_data

if __name__ == "__main__":
    extract_heart_disease_data()
    extract_data_to_csv()
    