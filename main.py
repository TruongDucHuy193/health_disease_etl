import argparse
import logging
from app.etl.extract.extract import extract_heart_disease_data, extract_data_to_csv
from app.etl.transform.transform import transform_heart_disease_data, merge_processed_files

def setup_logging():
    """Set up logging configuratio"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function for the ETL pipeline"""
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Heart Disease ETL Pipeline")
    parser.add_argument("--extract-only", action="store_true", help="Only perform data extraction")
    parser.add_argument("--transform-only", action="store_true", help="Only perform data transformation")
    parser.add_argument("--merge-files", action="store_true", help="Merge processed files into a single file")
    
    args = parser.parse_args()
    
    # Extract data if needed
    if not args.transform_only:
        logging.info("Starting data extraction...")
        extract_heart_disease_data()
        extract_data_to_csv()
        logging.info("Data extraction completed")
    
    # Transform data if needed
    if not args.extract_only:
        logging.info("Starting data transformation...")
        stats = transform_heart_disease_data()
        logging.info(f"Data transformation completed: {stats}")
    
    # Merge processed files if requested
    if args.merge_files:
        logging.info("Starting file merging...")
        merge_stats = merge_processed_files()
        logging.info(f"File merging completed: {merge_stats}")
    
    logging.info("ETL pipeline completed successfully")

if __name__ == "__main__":
    main()