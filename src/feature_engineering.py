# src/feature_engineering.py

import pandas as pd
import logging
from typing import Optional

# Use config for column names
from .config import INGREDIENTS_LIST_COLUMN, DIRECTIONS_LIST_COLUMN, TITLE_COLUMN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_recipe_features(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Adds features: number of ingredients, number of steps, and title word count.
    Uses column names defined in config.py.

    Args:
        df (Optional[pandas.DataFrame]): Input DataFrame with potential 'ingredients',
                                          'directions', 'title' list/string columns.

    Returns:
        Optional[pandas.DataFrame]: DataFrame with added feature columns ('num_ingredients',
                                      'num_steps', 'title_length'), or None if input df is None.
    """
    if df is None:
        logging.error("Input DataFrame is None. Cannot add features.")
        return None

    logging.info("Starting feature engineering...")
    # Work on a copy to avoid SettingWithCopyWarning if df is a slice
    df_out = df.copy()

    # Calculate number of ingredients (length of the list)
    ing_col = INGREDIENTS_LIST_COLUMN
    if ing_col in df_out.columns:
        # Check if item is a list before getting len, default to 0 otherwise
        df_out['num_ingredients'] = df_out[ing_col].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        logging.info(f"Added 'num_ingredients' feature from column '{ing_col}'.")
    else:
        logging.warning(f"Column '{ing_col}' not found. Skipping 'num_ingredients'.")

    # Calculate number of steps (length of the directions list)
    dir_col = DIRECTIONS_LIST_COLUMN
    if dir_col in df_out.columns:
        df_out['num_steps'] = df_out[dir_col].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        logging.info(f"Added 'num_steps' feature from column '{dir_col}'.")
    else:
        logging.warning(f"Column '{dir_col}' not found. Skipping 'num_steps'.")

    # Calculate title length (number of words)
    title_col = TITLE_COLUMN
    if title_col in df_out.columns:
        # Ensure title is string, handle NaNs, then split and count words
        df_out['title_length'] = df_out[title_col].fillna('').astype(str).apply(
            lambda x: len(x.split())
        )
        logging.info(f"Added 'title_length' feature from column '{title_col}'.")
    else:
        logging.warning(f"Column '{title_col}' not found. Skipping 'title_length'.")

    logging.info("Feature engineering complete.")

    # Log summary statistics of the newly created features
    feature_cols = ['num_ingredients', 'num_steps', 'title_length']
    existing_feature_cols = [col for col in feature_cols if col in df_out.columns]
    if existing_feature_cols:
        try:
            logging.info("\nSummary statistics for engineered features:")
            desc_string = df_out[existing_feature_cols].describe().to_string()
            for line in desc_string.split('\n'):
                 logging.info(line) # Log each line for better formatting in log files
        except Exception as e:
             logging.warning(f"Could not display summary statistics for features: {e}")

    return df_out