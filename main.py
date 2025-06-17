import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

from app.etl.extract.extract import extract_heart_disease_data, extract_data_to_csv
from app.etl.transform.transform import transform_raw_heart_disease_data
from app.etl.load.load_to_csv import load_to_csv, merge_processed_files

def setup_logging(verbose=False):
    """Set up logging configuration with optional verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function for the Heart Disease ETL pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Heart Disease ETL Pipeline")
    parser.add_argument("--extract-only", action="store_true", help="Only perform data extraction")
    parser.add_argument("--transform-only", action="store_true", help="Only perform data transformation")
    parser.add_argument("--merge-files", action="store_true", help="Merge processed files into a single file")
    parser.add_argument("--input-dir", type=str, default="app/data/raw/to_csv", 
                      help="Directory containing input CSV files")
    parser.add_argument("--output-dir", type=str, default="app/data/processed", 
                      help="Directory for processed files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--all", action="store_true", help="Run full ETL pipeline (extract, transform, merge)")
    parser.add_argument("--load-only", action="store_true", help="Only perform load operations")
    
    args = parser.parse_args()
    
    # Setup logging based on verbosity
    setup_logging(args.verbose)
    
    # If --all is specified, run everything
    if args.all:
        args.extract_only = False
        args.transform_only = False
        args.merge_files = True
        args.load_only = False
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Using output directory: {args.output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory: {e}")
        return 1
    
    # If no specific operation is requested or load-only is specified, run the full load process
    if (not args.extract_only and not args.transform_only and not args.merge_files and not args.all) or args.load_only:
        try:
            logging.info("Starting load process...")
              # Get transformed data - use new single file approach
            logging.info("Starting transformation process...")            # Transform the merged raw data - auto-create if missing
            input_file = os.path.join(args.input_dir, 'heart_disease_raw.csv')
            transformed_df, _ = transform_raw_heart_disease_data(input_file, args.output_dir)
            
            if not transformed_df.empty:
                transformed_data = {'processed_heart_disease.csv': transformed_df}
            else:
                logging.error("Transformation failed - no data to save")
                return 1
            
            # Load data to CSV files
            logging.info("\nLoading data to CSV files...")
            load_to_csv(transformed_data, args.output_dir)
              # Merge all processed files (output as processed_heart_disease.csv)
            logging.info("\nMerging all processed files...")
            merge_output = Path(args.output_dir) / "processed_heart_disease.csv"
            merge_stats = merge_processed_files(
                processed_directory=args.output_dir,
                output_file=str(merge_output)
            )
            logging.info(f"Merge statistics: {merge_stats}")
            
            logging.info("✅ Load process completed successfully")
            return 0
        except Exception as e:
            logging.error(f"❌ Load process failed: {e}")
            return 1
    
    # Extract data if needed
    if not args.transform_only and not args.load_only:
        try:
            logging.info("Starting data extraction...")
            extract_heart_disease_data()
            extract_data_to_csv()
            logging.info("✅ Data extraction completed successfully")
        except Exception as e:
            logging.error(f"❌ Data extraction failed: {e}")
            if not args.transform_only:  # Only exit if this was the only operation
                return 1
    
    # Transform data if needed
    transformed_dataframes = {}
    if not args.extract_only and not args.load_only:
        try:
            logging.info("Starting data transformation...")            # Transform the merged raw data - auto-create if missing
            input_file = os.path.join(args.input_dir, 'heart_disease_raw.csv')
            transformed_df, stats = transform_raw_heart_disease_data(input_file, args.output_dir)
            
            if not transformed_df.empty:
                transformed_dataframes = {'processed_heart_disease.csv': transformed_df}
            else:
                logging.error("Transformation failed - no data to save")
                if not args.transform_only:
                    return 1
                transformed_dataframes = {}
            
            # Save the transformed data using the load module
            if transformed_dataframes:
                logging.info("Saving transformed data to CSV files...")
                load_to_csv(transformed_dataframes, args.output_dir)
                logging.info(f"✅ Saved {len(transformed_dataframes)} files to {args.output_dir}")
            else:
                logging.warning("⚠️ No data to save after transformation")
            
            logging.info(f"✅ Data transformation completed with stats: {stats}")
        except Exception as e:
            logging.error(f"❌ Data transformation failed: {e}")
            if args.transform_only:  # Only exit if this was the only operation
                return 1
    
    # Merge processed files 
    if args.merge_files and not args.load_only:
        try:
            logging.info("Starting file merging...")
            merge_output = Path(args.output_dir) / "heart_disease_merge.csv"
            merge_stats = merge_processed_files(
                processed_directory=args.output_dir,
                output_file=str(merge_output)
            )
            
            if "error" in merge_stats:
                logging.warning(f"⚠️ File merging completed with warnings: {merge_stats['error']}")
            else:
                logging.info(f"✅ File merging completed: {merge_stats['files_merged']} files merged")
        except Exception as e:
            logging.error(f"❌ File merging failed: {e}")
            return 1
    
    logging.info("✅ ETL pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())