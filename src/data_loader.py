# src/data_loader.py

import pandas as pd
import logging
import time
import os
from typing import Optional

# Use relative import within the package
from .utils import parse_list
from .config import INGREDIENTS_LIST_COLUMN, DIRECTIONS_LIST_COLUMN, TITLE_COLUMN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads recipe data from a CSV file path.
    Cleans specified list-like columns by parsing string representations into Python lists.
    Drops 'Unnamed:' columns potentially created during CSV reads.

    Args:
        file_path (str): Local path to the CSV file.

    Returns:
        Optional[pandas.DataFrame]: Loaded and initially cleaned DataFrame, or None if loading fails.
    """
    df = None
    logging.info(f"Attempting to load dataset from local path: {file_path}")
    load_start_time = time.time()

    if not os.path.exists(file_path):
         logging.error(f"Data loading failed: File not found at {file_path}")
         return None

    try:
        df = pd.read_csv(file_path, low_memory=False)

        load_end_time = time.time()
        logging.info(f"Dataset loaded successfully in {load_end_time - load_start_time:.2f} seconds.")
        logging.info(f"Initial shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

        # Check for expected core columns
        required_cols = [INGREDIENTS_LIST_COLUMN, DIRECTIONS_LIST_COLUMN, TITLE_COLUMN]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             logging.warning(f"Missing expected columns: {missing_cols}. Subsequent steps might fail or be skipped.")

        # Attempt to parse string representations in list-like columns
        list_cols_to_parse = [INGREDIENTS_LIST_COLUMN, DIRECTIONS_LIST_COLUMN]
        logging.info(f"Parsing string representations in columns: {list_cols_to_parse}...")
        parse_start_time = time.time()

        for col in list_cols_to_parse:
            if col in df.columns:
                logging.debug(f"Checking column for parsing: {col}")
                # Check the type of the first few non-null elements to decide if parsing is needed
                needs_parsing = False
                try:
                    non_null_series = df[col].dropna()
                    if not non_null_series.empty:
                        first_valid_element = non_null_series.iloc[0]
                        # Check if it's a string that looks like a list representation "[...]"
                        if isinstance(first_valid_element, str) and first_valid_element.strip().startswith('['):
                            needs_parsing = True
                            logging.info(f"Column '{col}' appears to need parsing.")
                        else:
                             logging.info(f"Column '{col}' seems already parsed or not string lists. Skipping parsing.")
                    else:
                        logging.info(f"Column '{col}' contains only null values. Skipping parsing.")

                except IndexError:
                    logging.info(f"Column '{col}' has no non-null values to check. Skipping parsing.")
                except Exception as e:
                    logging.warning(f"Error checking column type for '{col}': {e}. Skipping parsing for this column.")

                if needs_parsing:
                    # Apply the safe parse_list function from utils
                    df[col] = df[col].apply(parse_list)
                    logging.info(f"Successfully applied parsing to column '{col}'.")
            else:
                # Column was missing, warning already issued above
                pass

        parse_end_time = time.time()
        logging.info(f"List parsing finished in {parse_end_time - parse_start_time:.2f} seconds.")

        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logging.info(f"Dropped unnamed columns: {unnamed_cols}")
            logging.info(f"DataFrame shape after dropping unnamed columns: {df.shape}")

        return df

    except FileNotFoundError:
        # This case is handled by the initial os.path.exists check, but kept for robustness
        logging.error(f"Local file not found at path: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading or initial cleaning: {e}", exc_info=True)
        return None