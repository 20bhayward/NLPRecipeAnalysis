# src/main.py

import pandas as pd
import logging
import os
import pickle
import time
import numpy as np
import sys
from typing import Dict, Any, Optional, Tuple

# Import project modules and configurations
from .config import (
    RECIPE_FILE, RESULTS_DIR, NETWORK_RESULTS_FILE, PROCESSED_DF_FILE, ALL_RESULTS_FILE,
    INGREDIENT_EMB_FILE, DIRECTION_EMB_FILE,
    TITLE_COLUMN, INGREDIENTS_LIST_COLUMN, DIRECTIONS_LIST_COLUMN,
    CLUSTER_TEXT_COLUMN_SPACY, CLUSTER_TEXT_COLUMN_SPACY_STR, TOPIC_TEXT_COLUMN_SPACY,
    INGREDIENTS_RAW_JOINED_COL, DIRECTIONS_RAW_JOINED_COL,
    KMEANS_TRAD_K_VALUES, KMEANS_TRAD_MAX_FEATURES, KMEANS_TRAD_MIN_DF, KMEANS_TRAD_MAX_DF, KMEANS_TRAD_N_COMPONENTS_SVD,
    LDA_TRAD_N_TOPICS, LDA_TRAD_MAX_FEATURES, LDA_TRAD_MIN_DF, LDA_TRAD_MAX_DF, LDA_TRAD_COHERENCE_MEASURE,
    TRANSFORMER_MODEL, INGREDIENTS_EMB_POOLING, DIRECTIONS_EMB_POOLING,
    INGREDIENTS_EMB_MAX_LEN, DIRECTIONS_EMB_MAX_LEN, EMBEDDING_BATCH_SIZE,
    KMEANS_EMB_K_VALUES, KMEANS_EMB_USE_UMAP, KMEANS_EMB_UMAP_NEIGHBORS,
    KMEANS_EMB_UMAP_MIN_DIST, KMEANS_EMB_UMAP_COMPONENTS, KMEANS_EMB_UMAP_METRIC,
    INGREDIENT_NETWORK_SAMPLE_SIZE, INGREDIENT_MIN_FREQ, INGREDIENT_MAX_FREQ, CO_OCCURRENCE_THRESHOLD,
    RANDOM_STATE
)
from .data_loader import load_and_clean_data
from .feature_engineering import add_recipe_features
from .utils import safe_join, preprocess_text_spacy, find_best_config, NLP as spacy_nlp_global # Import global NLP status check
from .clustering import (
    run_kmeans_ablation_tfidf, run_kmeans_ablation_embeddings,
    vectorize_text_tfidf, reduce_dimensions_svd, perform_kmeans,
    reduce_dimensions_umap, analyze_kmeans_stability, find_kmeans_outliers
)
from .topic_modeling import (
    run_lda_ablation, vectorize_text_count, perform_lda,
    analyze_lda_stability_coherence, find_lda_outliers
)
from .network_analysis import run_network_analysis_pipeline
from .embedding_generation import generate_transformer_embeddings

# --- Logging Setup ---
# Configure logging to write to a file and print to console
log_file_path = os.path.join(RESULTS_DIR, 'pipeline_run.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Global Control Flags ---
# Set these flags to True/False to enable/disable pipeline steps
RUN_DATA_LOADING_PREP = True
RUN_FEATURE_ENGINEERING = True
RUN_TRADITIONAL_PREPROCESSING = True # SpaCy step needed for traditional clustering/LDA
RUN_CLUSTERING_TRADITIONAL = True    # K-Means on TF-IDF/SVD
RUN_TOPIC_MODELING_TRADITIONAL = True # LDA on CountVec
RUN_NETWORK_ANALYSIS = True
RUN_TRANSFORMER_EMBEDDINGS = True    # Generate or Load Embeddings
RUN_CLUSTERING_TRANSFORMER = True    # K-Means on Embeddings
RUN_STABILITY_OUTLIER_ANALYSIS = True # Run stability/outlier checks on best models identified by ablation

# --- Helper function to prepare data for Stability/Outlier analysis ---
def get_data_for_analysis(df_analysis: pd.DataFrame, embeddings: Dict[str, np.ndarray], config: dict, analysis_type: str) -> Optional[Dict[str, Any]]:
    """
    Prepares the necessary data matrix or components for stability/outlier analysis
    based on the best configuration identified in ablation studies.

    Args:
        df_analysis: The main analysis DataFrame.
        embeddings: Dictionary possibly containing 'ingredients' or 'directions' embeddings.
        config: Dictionary containing the parameters of the best performing model configuration.
        analysis_type: String identifying the type of analysis ('kmeans_tfidf', 'lda', 'kmeans_embedding').

    Returns:
        Dictionary containing required data (e.g., 'data_for_kmeans', 'texts_tokenized')
        or None if preparation fails.
    """
    data_pack = {}
    try:
        if analysis_type == 'kmeans_tfidf':
            text_col = CLUSTER_TEXT_COLUMN_SPACY_STR
            if text_col not in df_analysis.columns: raise ValueError(f"Missing column: {text_col}")
            texts = df_analysis[text_col].fillna('').astype(str)
            # Re-vectorize and reduce dimensions using the best config parameters
            vectorizer, tfidf_matrix = vectorize_text_tfidf(texts, config['max_features'], config['min_df'], config['max_df'])
            if tfidf_matrix is None: raise ValueError("TF-IDF vectorization failed")
            svd_model, svd_embedding = reduce_dimensions_svd(tfidf_matrix, config['n_components'], RANDOM_STATE)
            # Use SVD embedding if available, otherwise fall back to TF-IDF matrix
            data_pack['data_for_kmeans'] = svd_embedding if svd_embedding is not None else tfidf_matrix
            data_pack['model_metric'] = 'cosine' # Metric typically used for TF-IDF/SVD space
            if data_pack['data_for_kmeans'] is None: raise ValueError("Data for KMeans is None after SVD/TFIDF")
            return data_pack

        elif analysis_type == 'lda':
            text_col = TOPIC_TEXT_COLUMN_SPACY
            if text_col not in df_analysis.columns: raise ValueError(f"Missing column: {text_col}")
             # Extract joined text strings and token lists from the tuple column
            processed_data = df_analysis[text_col].dropna().tolist()
            valid_data = [item for item in processed_data if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list)]
            texts_joined = [item[0] for item in valid_data]
            texts_tokenized = [item[1] for item in valid_data]
            # Filter out empty strings or token lists generated during preprocessing
            valid_indices = [i for i, (s, t) in enumerate(zip(texts_joined, texts_tokenized)) if s.strip() and t]
            if not valid_indices: raise ValueError("No valid texts/tokens found for LDA analysis.")
            # Ensure alignment between joined text and tokens
            texts_joined = [texts_joined[i] for i in valid_indices]
            texts_tokenized = [texts_tokenized[i] for i in valid_indices]

            data_pack['texts_joined'] = texts_joined # Needed for vectorization
            data_pack['texts_tokenized'] = texts_tokenized # Needed for coherence calculation
            data_pack['vectorizer_params'] = {'max_features': config['max_features'], 'min_df': config['min_df'], 'max_df': config['max_df']}
            data_pack['lda_params'] = {'n_topics': config['n_topics']}
            # Vectorize for input to LDA outlier analysis (needs document-topic matrix)
            vectorizer, count_matrix = vectorize_text_count(texts_joined, **data_pack['vectorizer_params'])
            if count_matrix is None: raise ValueError("Count Vectorization failed for LDA outlier data prep")
            data_pack['vectorized_data'] = count_matrix
            return data_pack

        elif analysis_type == 'kmeans_embedding':
            if 'ingredients' not in embeddings or embeddings['ingredients'] is None:
                raise ValueError("Ingredient embeddings not found or are None.")
            ingredient_embeddings = embeddings['ingredients']
            # Ensure embeddings match the current DataFrame length (important if sampling occurred)
            if ingredient_embeddings.shape[0] != len(df_analysis):
                 raise ValueError(f"Embedding length ({ingredient_embeddings.shape[0]}) != DataFrame length ({len(df_analysis)})")

            data_pack['embeddings'] = ingredient_embeddings
            data_pack['original_texts'] = df_analysis[INGREDIENTS_RAW_JOINED_COL] # Needed for outlier text context

            # Apply UMAP if it was part of the best configuration
            if config.get('use_umap', False):
                umap_params = {'n_neighbors': config['umap_n_neighbors'], 'min_dist': config['umap_min_dist'],
                               'n_components': config['umap_n_components'], 'metric': config['umap_metric']}
                umap_model, umap_embedding = reduce_dimensions_umap(data_pack['embeddings'], **umap_params, random_state=RANDOM_STATE)
                if umap_embedding is None: raise ValueError("UMAP failed for best embedding config")
                data_pack['data_for_kmeans'] = umap_embedding
                data_pack['model_metric'] = 'euclidean' # Metric typically used for UMAP space
            else:
                data_pack['data_for_kmeans'] = data_pack['embeddings'] # Use raw embeddings
                data_pack['model_metric'] = 'cosine' # Metric typically used for raw embeddings

            if data_pack['data_for_kmeans'] is None: raise ValueError("Data for KMeans (Embeddings) is None")
            return data_pack

        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")

    except Exception as e:
        logging.error(f"Error preparing data for {analysis_type} stability/outlier analysis: {e}", exc_info=True)
        return None


def run_pipeline(sample_size: Optional[int] = 50000):
    """Runs the main recipe analysis pipeline."""
    logging.info("=============================================")
    logging.info("=== Starting Recipe NLP Analysis Pipeline ===")
    logging.info("=============================================")
    pipeline_start_time = time.time()

    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR)
            logging.info(f"Created results directory: {RESULTS_DIR}")
        except OSError as e:
            logging.error(f"Failed to create results directory {RESULTS_DIR}: {e}. Output saving will fail.", exc_info=True)
            return

    # --- Step Execution Permissions ---
    # These flags track if preconditions for steps are met 
    can_run_feature_eng = True
    can_run_traditional_preprocessing = True
    can_run_clustering_trad = True
    can_run_topic_modeling_trad = True
    can_run_embeddings_ing = True
    can_run_embeddings_dir = True
    can_run_clustering_transformer = True
    can_run_network = True
    can_run_stability_outlier = True

    # --- Result Containers ---
    df_analysis = None
    # Store all configuration flags and results in this dictionary
    all_results = {'pipeline_config': {flag: globals()[flag] for flag in globals() if flag.startswith('RUN_')}}
    all_results['sample_size_used'] = sample_size
    # Store loaded or generated embeddings here (will not be saved in the main results pickle)
    generated_embeddings = {'ingredients': None, 'directions': None}

    # === 1. Load Data and Initial Preparation ===
    if RUN_DATA_LOADING_PREP:
        logging.info("--- 1. Data Loading and Initial Preparation ---")
        df = load_and_clean_data(file_path=RECIPE_FILE)
        if df is None:
            logging.error("Pipeline halted: Data loading failed.")
            return

        # Apply sampling if specified
        if sample_size is not None and sample_size > 0 and sample_size < len(df):
            logging.info(f"Sampling {sample_size} recipes for analysis (random_state={RANDOM_STATE}).")
            try:
                df_analysis = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
            except ValueError as e:
                 logging.error(f"Sampling failed ({e}). Check sample size. Aborting.")
                 return
        elif sample_size is not None and sample_size >= len(df):
             logging.info(f"Sample size {sample_size} >= dataset size {len(df)}. Using full loaded dataset.")
             df_analysis = df.copy()
        else: # sample_size is None or 0
             logging.info("Using full loaded dataset for analysis.")
             df_analysis = df.copy()

        df_analysis.reset_index(drop=True, inplace=True) # Ensure consistent index after sampling
        logging.info(f"DataFrame shape for analysis: {df_analysis.shape}")
        all_results['data_shape_analysis'] = df_analysis.shape

        # Create raw joined text columns (required for Transformer embeddings)
        logging.info("Creating joined raw text columns for transformer input...")
        if INGREDIENTS_LIST_COLUMN in df_analysis.columns:
            df_analysis[INGREDIENTS_RAW_JOINED_COL] = df_analysis[INGREDIENTS_LIST_COLUMN].apply(safe_join)
            logging.info(f"Created '{INGREDIENTS_RAW_JOINED_COL}'.")
        else:
            logging.warning(f"Cannot create '{INGREDIENTS_RAW_JOINED_COL}': Column '{INGREDIENTS_LIST_COLUMN}' missing.")
            can_run_embeddings_ing = False
            can_run_clustering_transformer = False # Depends on ingredient embeddings
            can_run_network = False # Network analysis also uses ingredients list
            can_run_feature_eng = False # num_ingredients depends on this column

        if DIRECTIONS_LIST_COLUMN in df_analysis.columns:
            df_analysis[DIRECTIONS_RAW_JOINED_COL] = df_analysis[DIRECTIONS_LIST_COLUMN].apply(safe_join)
            logging.info(f"Created '{DIRECTIONS_RAW_JOINED_COL}'.")
        else:
            logging.warning(f"Cannot create '{DIRECTIONS_RAW_JOINED_COL}': Column '{DIRECTIONS_LIST_COLUMN}' missing.")
            can_run_embeddings_dir = False
            can_run_topic_modeling_trad = False # LDA needs directions text
            can_run_feature_eng = False # num_steps depends on this column

    else:
        logging.info("Skipping Data Loading step.")
        return # Cannot proceed without data

    # === 2. Feature Engineering ===
    if RUN_FEATURE_ENGINEERING and can_run_feature_eng:
        logging.info("--- 2. Feature Engineering ---")
        df_analysis = add_recipe_features(df_analysis)
        if df_analysis is None:
            logging.error("Pipeline halted: Feature engineering failed unexpectedly.")
            return
    elif RUN_FEATURE_ENGINEERING:
        logging.warning("Skipping Feature Engineering: Preconditions not met (required columns missing).")

    # === 3. Traditional Preprocessing (SpaCy) ===
    # This step prepares text data (lemmatization, stop word removal) for TF-IDF/CountVec methods
    if RUN_TRADITIONAL_PREPROCESSING:
        logging.info("--- 3. Traditional Preprocessing (SpaCy) ---")
        preprocessing_start_time = time.time()

        # Check SpaCy model status (load attempt happens within preprocess_text_spacy)
        _, _ = preprocess_text_spacy("Initialize spacy check.") # Call once to trigger load/check
        if spacy_nlp_global == "Failed":
            logging.error("SpaCy model failed to load. Disabling traditional clustering and topic modeling.")
            can_run_traditional_preprocessing = False
            can_run_clustering_trad = False
            can_run_topic_modeling_trad = False
        else:
            # Preprocess ingredients for K-Means TF-IDF/SVD
            if can_run_clustering_trad and INGREDIENTS_LIST_COLUMN in df_analysis.columns:
                logging.info(f"Preprocessing ingredients ('{INGREDIENTS_LIST_COLUMN}') for TF-IDF clustering...")
                # Use raw joined text if available, otherwise join the list column
                source_col = INGREDIENTS_RAW_JOINED_COL if INGREDIENTS_RAW_JOINED_COL in df_analysis.columns else INGREDIENTS_LIST_COLUMN
                texts_to_process = df_analysis[source_col].apply(safe_join) if source_col == INGREDIENTS_LIST_COLUMN else df_analysis[source_col]

                # Apply SpaCy preprocessing: returns tuple (joined_text, token_list)
                df_analysis[CLUSTER_TEXT_COLUMN_SPACY] = texts_to_process.apply(preprocess_text_spacy)

                # Create a separate column with only the processed string for vectorizers
                processed_texts_clust = df_analysis[CLUSTER_TEXT_COLUMN_SPACY].apply(lambda x: x[0] if isinstance(x, tuple) else "")
                if processed_texts_clust.str.strip().eq('').all():
                    logging.warning(f"SpaCy preprocessing of ingredients yielded empty strings. Disabling traditional clustering.")
                    can_run_clustering_trad = False
                    df_analysis = df_analysis.drop(columns=[CLUSTER_TEXT_COLUMN_SPACY], errors='ignore')
                else:
                    df_analysis[CLUSTER_TEXT_COLUMN_SPACY_STR] = processed_texts_clust
                    logging.info(f"Created '{CLUSTER_TEXT_COLUMN_SPACY}' (tuple) and '{CLUSTER_TEXT_COLUMN_SPACY_STR}' (string) columns.")
            elif can_run_clustering_trad:
                 logging.warning(f"Cannot preprocess ingredients for TF-IDF: Column '{INGREDIENTS_LIST_COLUMN}' missing.")
                 can_run_clustering_trad = False

            # Preprocess directions for LDA
            if can_run_topic_modeling_trad and DIRECTIONS_LIST_COLUMN in df_analysis.columns:
                logging.info(f"Preprocessing directions ('{DIRECTIONS_LIST_COLUMN}') for LDA...")
                source_col = DIRECTIONS_RAW_JOINED_COL if DIRECTIONS_RAW_JOINED_COL in df_analysis.columns else DIRECTIONS_LIST_COLUMN
                texts_to_process = df_analysis[source_col].apply(safe_join) if source_col == DIRECTIONS_LIST_COLUMN else df_analysis[source_col]

                # Apply SpaCy preprocessing: returns tuple (joined_text, token_list)
                df_analysis[TOPIC_TEXT_COLUMN_SPACY] = texts_to_process.apply(preprocess_text_spacy)

                # Check if preprocessing yielded results
                processed_texts_topic = df_analysis[TOPIC_TEXT_COLUMN_SPACY].apply(lambda x: x[0] if isinstance(x, tuple) else "")
                if processed_texts_topic.str.strip().eq('').all():
                     logging.warning(f"SpaCy preprocessing of directions yielded empty results. Disabling traditional topic modeling.")
                     can_run_topic_modeling_trad = False
                     df_analysis = df_analysis.drop(columns=[TOPIC_TEXT_COLUMN_SPACY], errors='ignore')
                else:
                     # LDA needs the tuple (text, tokens) - tokens for coherence, text for vectorization
                     logging.info(f"Created '{TOPIC_TEXT_COLUMN_SPACY}' (tuple) column for LDA.")
            elif can_run_topic_modeling_trad:
                 logging.warning(f"Cannot preprocess directions for LDA: Column '{DIRECTIONS_LIST_COLUMN}' missing.")
                 can_run_topic_modeling_trad = False

        logging.info(f"Traditional preprocessing step took {time.time() - preprocessing_start_time:.2f} seconds.")

    elif RUN_CLUSTERING_TRADITIONAL or RUN_TOPIC_MODELING_TRADITIONAL:
        # Disable dependent steps if preprocessing is skipped but they were enabled
        logging.warning("Skipping Traditional Preprocessing, but Traditional Clustering/Topic Modeling was enabled. Disabling them.")
        can_run_clustering_trad = False
        can_run_topic_modeling_trad = False
    else:
        logging.info("Skipping Traditional Preprocessing step.")


    # === 4. Traditional Clustering (TF-IDF/SVD K-Means) ===
    if RUN_CLUSTERING_TRADITIONAL and can_run_clustering_trad:
        if CLUSTER_TEXT_COLUMN_SPACY_STR in df_analysis.columns:
            logging.info("--- 4. Running K-Means Ablation Study (TF-IDF/SVD) ---")
            kmeans_tfidf_results = run_kmeans_ablation_tfidf(
                df=df_analysis, text_column=CLUSTER_TEXT_COLUMN_SPACY_STR,
                k_values=KMEANS_TRAD_K_VALUES, max_features_list=KMEANS_TRAD_MAX_FEATURES,
                min_df_list=KMEANS_TRAD_MIN_DF, max_df_list=KMEANS_TRAD_MAX_DF,
                n_components_list=KMEANS_TRAD_N_COMPONENTS_SVD, random_state=RANDOM_STATE)
            all_results['kmeans_ablation_tfidf'] = kmeans_tfidf_results
        else:
             logging.warning(f"Skipping K-Means (TF-IDF): Input column '{CLUSTER_TEXT_COLUMN_SPACY_STR}' not found.")
    elif RUN_CLUSTERING_TRADITIONAL:
        logging.info("Skipping K-Means (TF-IDF): Preconditions not met (SpaCy failed or column missing).")


    # === 5. Traditional Topic Modeling (LDA) ===
    if RUN_TOPIC_MODELING_TRADITIONAL and can_run_topic_modeling_trad:
        if TOPIC_TEXT_COLUMN_SPACY in df_analysis.columns:
            logging.info("--- 5. Running LDA Ablation Study (CountVec) ---")
            lda_results = run_lda_ablation(
                df=df_analysis, processed_text_column=TOPIC_TEXT_COLUMN_SPACY, # Pass the tuple column
                topic_values=LDA_TRAD_N_TOPICS, max_features_list=LDA_TRAD_MAX_FEATURES,
                min_df_list=LDA_TRAD_MIN_DF, max_df_list=LDA_TRAD_MAX_DF, random_state=RANDOM_STATE)
            all_results['lda_ablation_countvec'] = lda_results
        else:
             logging.warning(f"Skipping LDA (CountVec): Input column '{TOPIC_TEXT_COLUMN_SPACY}' not found.")
    elif RUN_TOPIC_MODELING_TRADITIONAL:
        logging.info("Skipping LDA (CountVec): Preconditions not met (SpaCy failed or column missing).")


    # === 6. Generate or Load Transformer Embeddings ===
    # Runs if flag is True OR if needed for downstream embedding clustering
    should_generate_or_load = RUN_TRANSFORMER_EMBEDDINGS or (RUN_CLUSTERING_TRANSFORMER and can_run_clustering_transformer)
    if should_generate_or_load:
        logging.info("--- 6. Checking/Generating Transformer Embeddings ---")
        embedding_info = {'generation_params': {
                             'model': TRANSFORMER_MODEL, 'batch_size': EMBEDDING_BATCH_SIZE,
                             'ing_max_len': INGREDIENTS_EMB_MAX_LEN, 'dir_max_len': DIRECTIONS_EMB_MAX_LEN,
                             'ing_pooling': INGREDIENTS_EMB_POOLING, 'dir_pooling': DIRECTIONS_EMB_POOLING }}

        # --- Ingredients Embeddings ---
        if can_run_embeddings_ing:
            loaded_ing_ok = False
            # Try loading first if generation isn't forced and file exists
            if not RUN_TRANSFORMER_EMBEDDINGS and os.path.exists(INGREDIENT_EMB_FILE):
                logging.info(f"Attempting to load existing ingredient embeddings from {INGREDIENT_EMB_FILE}")
                try:
                    loaded_emb = np.load(INGREDIENT_EMB_FILE)
                    # Validate shape against current DataFrame size
                    if loaded_emb.shape[0] == len(df_analysis):
                        generated_embeddings['ingredients'] = loaded_emb
                        embedding_info['ingredient_emb_source'] = 'Loaded from file'
                        embedding_info['ingredient_emb_shape'] = loaded_emb.shape
                        logging.info(f"Loaded ingredient embeddings. Shape: {loaded_emb.shape}")
                        loaded_ing_ok = True
                    else:
                        logging.warning(f"Loaded ingredient embedding rows ({loaded_emb.shape[0]}) != DataFrame rows ({len(df_analysis)}). Will regenerate if generation flag is set.")
                except Exception as e:
                    logging.warning(f"Failed to load existing ingredient embeddings from {INGREDIENT_EMB_FILE}: {e}. Will regenerate if generation flag is set.")

            # Generate if not loaded successfully OR if forced by RUN_TRANSFORMER_EMBEDDINGS flag
            if not loaded_ing_ok and RUN_TRANSFORMER_EMBEDDINGS:
                 logging.info(f"Generating embeddings for ingredients ('{INGREDIENTS_RAW_JOINED_COL}') using '{TRANSFORMER_MODEL}'...")
                 if INGREDIENTS_RAW_JOINED_COL not in df_analysis.columns:
                     logging.error("Cannot generate ingredient embeddings: Raw joined text column missing.")
                     can_run_clustering_transformer = False # Cannot cluster without embeddings
                 else:
                     ingredient_embeddings = generate_transformer_embeddings(
                         texts=df_analysis[INGREDIENTS_RAW_JOINED_COL], model_name=TRANSFORMER_MODEL,
                         batch_size=EMBEDDING_BATCH_SIZE, max_length=INGREDIENTS_EMB_MAX_LEN,
                         pooling_strategy=INGREDIENTS_EMB_POOLING)

                     # Validate generated embeddings
                     if ingredient_embeddings is not None and ingredient_embeddings.shape[0] == len(df_analysis):
                          generated_embeddings['ingredients'] = ingredient_embeddings
                          embedding_info['ingredient_emb_source'] = 'Generated'
                          embedding_info['ingredient_emb_shape'] = ingredient_embeddings.shape
                          # Save generated embeddings
                          try:
                              np.save(INGREDIENT_EMB_FILE, ingredient_embeddings)
                              logging.info(f"Ingredient embeddings saved to {INGREDIENT_EMB_FILE}")
                          except Exception as e:
                              logging.error(f"Failed to save generated ingredient embeddings: {e}")
                     elif ingredient_embeddings is not None: # Shape mismatch after generation
                          logging.error(f"Generated ingredient embedding rows ({ingredient_embeddings.shape[0]}) != DataFrame rows ({len(df_analysis)})! Disabling embedding clustering.")
                          can_run_clustering_transformer = False
                     else: # Generation failed completely
                          logging.error("Failed to generate ingredient embeddings. Disabling embedding clustering.")
                          can_run_clustering_transformer = False
            elif not loaded_ing_ok and not RUN_TRANSFORMER_EMBEDDINGS:
                 logging.warning("Ingredient embeddings need to be generated but RUN_TRANSFORMER_EMBEDDINGS is False. Embedding clustering will be disabled.")
                 can_run_clustering_transformer = False

        else: # Preconditions for ingredients embedding not met
            logging.warning("Skipping ingredient embedding generation/loading: Preconditions not met.")
            can_run_clustering_transformer = False

        # --- Directions Embeddings ---
        if can_run_embeddings_dir:
            loaded_dir_ok = False
            if not RUN_TRANSFORMER_EMBEDDINGS and os.path.exists(DIRECTION_EMB_FILE):
                 logging.info(f"Attempting to load existing direction embeddings from {DIRECTION_EMB_FILE}")
                 try:
                     loaded_emb = np.load(DIRECTION_EMB_FILE)
                     if loaded_emb.shape[0] == len(df_analysis):
                         generated_embeddings['directions'] = loaded_emb
                         embedding_info['direction_emb_source'] = 'Loaded from file'
                         embedding_info['direction_emb_shape'] = loaded_emb.shape
                         logging.info(f"Loaded direction embeddings. Shape: {loaded_emb.shape}")
                         loaded_dir_ok = True
                     else:
                          logging.warning(f"Loaded direction embedding rows ({loaded_emb.shape[0]}) != DataFrame rows ({len(df_analysis)}). Will regenerate if flag set.")
                 except Exception as e:
                      logging.warning(f"Failed to load existing direction embeddings from {DIRECTION_EMB_FILE}: {e}. Will regenerate if flag set.")

            if not loaded_dir_ok and RUN_TRANSFORMER_EMBEDDINGS:
                  logging.info(f"Generating embeddings for directions ('{DIRECTIONS_RAW_JOINED_COL}') using '{TRANSFORMER_MODEL}'...")
                  if DIRECTIONS_RAW_JOINED_COL not in df_analysis.columns:
                      logging.error("Cannot generate direction embeddings: Raw joined text column missing.")
                  else:
                      direction_embeddings = generate_transformer_embeddings(
                          texts=df_analysis[DIRECTIONS_RAW_JOINED_COL], model_name=TRANSFORMER_MODEL,
                          batch_size=EMBEDDING_BATCH_SIZE, max_length=DIRECTIONS_EMB_MAX_LEN,
                          pooling_strategy=DIRECTIONS_EMB_POOLING)

                      if direction_embeddings is not None and direction_embeddings.shape[0] == len(df_analysis):
                           generated_embeddings['directions'] = direction_embeddings
                           embedding_info['direction_emb_source'] = 'Generated'
                           embedding_info['direction_emb_shape'] = direction_embeddings.shape
                           try:
                               np.save(DIRECTION_EMB_FILE, direction_embeddings)
                               logging.info(f"Direction embeddings saved to {DIRECTION_EMB_FILE}")
                           except Exception as e:
                               logging.error(f"Failed to save generated direction embeddings: {e}")
                      elif direction_embeddings is not None:
                           logging.error(f"Generated direction embedding rows ({direction_embeddings.shape[0]}) != DataFrame rows ({len(df_analysis)})!")
                      else:
                           logging.error("Failed to generate direction embeddings.")
        else:
             logging.warning("Skipping direction embedding generation/loading: Preconditions not met.")

        all_results['transformer_embeddings_info'] = embedding_info # Store metadata about embeddings
    else:
        logging.info("Skipping Transformer Embedding Generation/Loading step.")


    # === 7. Clustering on Transformer Embeddings ===
    if RUN_CLUSTERING_TRANSFORMER and can_run_clustering_transformer:
        # Final check: ensure embeddings are actually available and match DataFrame
        if generated_embeddings['ingredients'] is not None and generated_embeddings['ingredients'].shape[0] == len(df_analysis) and INGREDIENTS_RAW_JOINED_COL in df_analysis.columns:
            logging.info("--- 7. Running K-Means Ablation Study (Transformer Embeddings) ---")
            kmeans_emb_results = run_kmeans_ablation_embeddings(
                 embeddings=generated_embeddings['ingredients'],
                 original_texts=df_analysis[INGREDIENTS_RAW_JOINED_COL],
                 k_values=KMEANS_EMB_K_VALUES,
                 use_umap=KMEANS_EMB_USE_UMAP,
                 umap_neighbors_list=KMEANS_EMB_UMAP_NEIGHBORS,
                 umap_min_dist_list=KMEANS_EMB_UMAP_MIN_DIST,
                 umap_components_list=KMEANS_EMB_UMAP_COMPONENTS,
                 umap_metric=KMEANS_EMB_UMAP_METRIC,
                 random_state=RANDOM_STATE
             )
            all_results['kmeans_ablation_embeddings'] = kmeans_emb_results
        else:
             logging.warning("Skipping K-Means (Embeddings): Ingredient embeddings not available, shape mismatch, or raw text column missing.")
    elif RUN_CLUSTERING_TRANSFORMER:
        logging.info("Skipping Clustering on Transformer Embeddings: Preconditions not met.")


    # === 8. Network Analysis ===
    if RUN_NETWORK_ANALYSIS and can_run_network:
        if INGREDIENTS_LIST_COLUMN in df_analysis.columns:
            logging.info("--- 8. Running Network Analysis Step ---")
            network_analysis_results = run_network_analysis_pipeline(
                df=df_analysis, ingredient_col=INGREDIENTS_LIST_COLUMN,
                sample_size=INGREDIENT_NETWORK_SAMPLE_SIZE,
                min_freq=INGREDIENT_MIN_FREQ, max_freq=INGREDIENT_MAX_FREQ,
                co_occurrence_threshold=CO_OCCURRENCE_THRESHOLD,
                random_state=RANDOM_STATE)

            all_results['network_analysis'] = network_analysis_results # Store results

            # Save network results
            if network_analysis_results and network_analysis_results.get('graph') is not None:
                try:
                    # Exclude the graph object itself to keep pickle file smaller
                    results_to_save = {k: v for k, v in network_analysis_results.items() if k != 'graph'}
                    results_to_save['graph_summary'] = network_analysis_results.get('graph_summary', {}) # Ensure summary is saved
                    with open(NETWORK_RESULTS_FILE, 'wb') as f:
                        pickle.dump(results_to_save, f)
                    logging.info(f"Network analysis results (excluding graph object) saved to {NETWORK_RESULTS_FILE}")
                except Exception as e:
                    logging.error(f"Could not save network analysis results: {e}", exc_info=True)
            elif network_analysis_results and network_analysis_results.get('error'):
                 logging.warning(f"Network analysis completed with error: {network_analysis_results.get('error')}")
            elif not network_analysis_results:
                 logging.warning("Network analysis did not return any results.")

        else:
            logging.warning(f"Skipping Network Analysis: Required column '{INGREDIENTS_LIST_COLUMN}' not found.")

    elif RUN_NETWORK_ANALYSIS:
        logging.info("Skipping Network Analysis: Preconditions not met (column missing).")


    # === 9. Stability and Outlier Analysis ===
    if RUN_STABILITY_OUTLIER_ANALYSIS and can_run_stability_outlier:
        logging.info("--- 9. Running Stability and Outlier Analysis ---")
        analysis_time_start = time.time()
        stability_results = {}
        outlier_results = {}

        # --- Analyze Best K-Means (TF-IDF/SVD) Model ---
        if 'kmeans_ablation_tfidf' in all_results and all_results['kmeans_ablation_tfidf']:
            logging.info("--- 9a. Analyzing Best K-Means (TF-IDF/SVD) Model ---")
            # Find the best config based on silhouette score
            best_config_key = find_best_config(all_results['kmeans_ablation_tfidf'], 'silhouette_score', higher_is_better=True)
            if best_config_key:
                best_config_params = all_results['kmeans_ablation_tfidf'][best_config_key]
                # Prepare the specific data needed for this best config
                data_pack = get_data_for_analysis(df_analysis, {}, best_config_params, 'kmeans_tfidf')
                if data_pack:
                    # Re-run KMeans on the full data with the best parameters
                    best_kmeans_model, _ = perform_kmeans(data_pack['data_for_kmeans'], n_clusters=best_config_params['k'], random_state=RANDOM_STATE)
                    if best_kmeans_model:
                        # Analyze Stability (ARI)
                        stability_score = analyze_kmeans_stability(data_pack['data_for_kmeans'], n_clusters=best_config_params['k'], random_state=RANDOM_STATE)
                        stability_results['kmeans_tfidf'] = {'best_config': best_config_key, 'avg_ari': stability_score}
                        # Find Outliers (distance to centroid)
                        outlier_indices, outlier_distances = find_kmeans_outliers(data_pack['data_for_kmeans'], best_kmeans_model, metric=data_pack['model_metric'])
                        if outlier_indices is not None:
                            outlier_texts = df_analysis.loc[outlier_indices, CLUSTER_TEXT_COLUMN_SPACY_STR].tolist() if CLUSTER_TEXT_COLUMN_SPACY_STR in df_analysis else ["Text unavailable"] * len(outlier_indices)
                            outlier_results['kmeans_tfidf'] = {'best_config': best_config_key, 'indices': outlier_indices.tolist(), 'distances': outlier_distances.tolist(), 'texts': outlier_texts}
                    else: logging.warning("Could not re-run best KMeans TF-IDF model for stability/outlier analysis.")
                else: logging.warning("Could not prepare data for KMeans TF-IDF stability/outlier analysis.")
            else: logging.warning("Could not find best configuration for KMeans TF-IDF analysis.")
        else: logging.info("Skipping KMeans TF-IDF stability/outlier analysis: No ablation results found.")

        # --- Analyze Best LDA Model ---
        if 'lda_ablation_countvec' in all_results and all_results['lda_ablation_countvec']:
             logging.info("--- 9b. Analyzing Best LDA Model ---")
             metric_key = f'coherence_{LDA_TRAD_COHERENCE_MEASURE}'
             best_config_key = find_best_config(all_results['lda_ablation_countvec'], metric_key, higher_is_better=True)
             if best_config_key:
                 best_config_params = all_results['lda_ablation_countvec'][best_config_key]
                 data_pack = get_data_for_analysis(df_analysis, {}, best_config_params, 'lda')
                 if data_pack:
                     # Analyze Stability (Coherence on subsamples)
                     mean_coh, std_coh = analyze_lda_stability_coherence(
                         texts_joined=data_pack['texts_joined'], texts_tokenized=data_pack['texts_tokenized'],
                         vectorizer_params=data_pack['vectorizer_params'], lda_params=data_pack['lda_params'],
                         random_state=RANDOM_STATE)
                     stability_results['lda'] = {'best_config': best_config_key, f'mean_coherence_{LDA_TRAD_COHERENCE_MEASURE}': mean_coh, f'std_coherence_{LDA_TRAD_COHERENCE_MEASURE}': std_coh}

                     # Find Outliers (low max topic probability)
                     # Re-run LDA on the full (filtered) data prepared in data_pack
                     best_lda_model = perform_lda(data_pack['vectorized_data'], random_state=RANDOM_STATE, **data_pack['lda_params'])
                     if best_lda_model:
                          try:
                              # Get document-topic distributions
                              doc_topic_matrix = best_lda_model.transform(data_pack['vectorized_data'])
                              outlier_indices, outlier_probs = find_lda_outliers(doc_topic_matrix)
                              if outlier_indices is not None:
                                   # Map outlier indices back to original DataFrame texts if possible
                                   original_texts_lda = df_analysis.iloc[outlier_indices][TOPIC_TEXT_COLUMN_SPACY].apply(lambda x: x[0] if isinstance(x, tuple) else "").tolist() if TOPIC_TEXT_COLUMN_SPACY in df_analysis else ["Text unavailable"] * len(outlier_indices)
                                   outlier_results['lda'] = {'best_config': best_config_key, 'indices': outlier_indices.tolist(), 'max_probabilities': outlier_probs.tolist(), 'texts': original_texts_lda}
                          except Exception as transform_err:
                               logging.error(f"Failed to get document-topic matrix for LDA outlier analysis: {transform_err}")
                     else: logging.warning("Could not re-run best LDA model for outlier analysis.")
                 else: logging.warning("Could not prepare data for LDA stability/outlier analysis.")
             else: logging.warning("Could not find best configuration for LDA analysis.")
        else: logging.info("Skipping LDA stability/outlier analysis: No ablation results found.")

        # --- Analyze Best K-Means (Embedding) Model ---
        # Check if embeddings were generated/loaded and results exist
        embeddings_available = generated_embeddings.get('ingredients') is not None
        results_available = 'kmeans_ablation_embeddings' in all_results and all_results['kmeans_ablation_embeddings']

        if results_available and embeddings_available:
            logging.info("--- 9c. Analyzing Best K-Means (Embeddings) Model ---")
            best_config_key = find_best_config(all_results['kmeans_ablation_embeddings'], 'silhouette_score', higher_is_better=True)
            if best_config_key:
                best_config_params = all_results['kmeans_ablation_embeddings'][best_config_key]
                # Ensure necessary params are present (should be stored during ablation)
                if 'use_umap' not in best_config_params: best_config_params['use_umap'] = KMEANS_EMB_USE_UMAP # Fallback if missing
                if not isinstance(best_config_params.get('use_umap'), bool): best_config_params['use_umap'] = KMEANS_EMB_USE_UMAP

                # Prepare data, potentially applying UMAP again
                data_pack = get_data_for_analysis(df_analysis, generated_embeddings, best_config_params, 'kmeans_embedding')

                if data_pack:
                    # Re-run KMeans with best config
                    best_kmeans_model, _ = perform_kmeans(data_pack['data_for_kmeans'], n_clusters=best_config_params['k'], random_state=RANDOM_STATE)
                    if best_kmeans_model:
                        # Analyze Stability (ARI)
                        stability_score = analyze_kmeans_stability(data_pack['data_for_kmeans'], n_clusters=best_config_params['k'], random_state=RANDOM_STATE)
                        stability_results['kmeans_embedding'] = {'best_config': best_config_key, 'avg_ari': stability_score}
                        # Find Outliers (distance to centroid)
                        outlier_indices, outlier_distances = find_kmeans_outliers(data_pack['data_for_kmeans'], best_kmeans_model, metric=data_pack['model_metric'])
                        if outlier_indices is not None:
                            # Get original texts corresponding to outliers
                            outlier_texts = data_pack['original_texts'].iloc[outlier_indices].tolist()
                            outlier_results['kmeans_embedding'] = {'best_config': best_config_key, 'indices': outlier_indices.tolist(), 'distances': outlier_distances.tolist(), 'texts': outlier_texts}
                    else: logging.warning("Could not re-run best KMeans Embedding model for analysis.")
                else: logging.warning("Could not prepare data for KMeans Embedding stability/outlier analysis.")
            else: logging.warning("Could not find best configuration for KMeans Embedding analysis.")
        elif RUN_STABILITY_OUTLIER_ANALYSIS: # Only log if the step was intended to run
             logging.info("Skipping KMeans Embedding stability/outlier analysis: No results or embeddings found.")

        # Add stability/outlier results to main results dictionary
        all_results['stability_analysis'] = stability_results
        all_results['outlier_analysis'] = outlier_results
        logging.info(f"--- Finished Stability and Outlier Analysis (Took {time.time() - analysis_time_start:.2f}s) ---")

    else:
        logging.info("Skipping Stability and Outlier Analysis step.")


    # === 10. Save Processed DataFrame and Final Results ===
    # Save the final DataFrame
    logging.info(f"Attempting to save analysis DataFrame to {PROCESSED_DF_FILE}...")
    try:
        # Exclude columns that might be problematic for pickle or extremely large if necessary
        cols_to_exclude = []
        df_to_save = df_analysis.drop(columns=cols_to_exclude, errors='ignore')
        df_to_save.to_pickle(PROCESSED_DF_FILE)
        logging.info(f"Analysis DataFrame saved successfully ({PROCESSED_DF_FILE}).")
    except Exception as e:
        logging.error(f"Failed to save analysis DataFrame: {e}", exc_info=True)

    # Save the dictionary containing all ablation results, model parameters, stability/outlier info etc.
    logging.info(f"Attempting to save all analysis results to {ALL_RESULTS_FILE}...")
    try:
       with open(ALL_RESULTS_FILE, 'wb') as f:
           pickle.dump(all_results, f)
       logging.info(f"All analysis results saved successfully ({ALL_RESULTS_FILE}).")
    except Exception as e:
       logging.error(f"Could not save all analysis results to pickle file: {e}", exc_info=True)

    # === Pipeline End ===
    pipeline_end_time = time.time()
    logging.info("===============================================")
    logging.info(f"=== Recipe NLP Analysis Pipeline Finished ===")
    logging.info(f"=== Total time: {pipeline_end_time - pipeline_start_time:.2f} seconds ===")
    logging.info(f"=== Results saved in: {RESULTS_DIR} ===")
    logging.info(f"=== Log file: {log_file_path} ===")
    logging.info("===============================================")

if __name__ == "__main__":
    # Set a sample size for development/testing, or None to use the full dataset
    # SAMPLE_SIZE_MAIN = 5000
    SAMPLE_SIZE_MAIN = 50000 # Moderate sample
    # SAMPLE_SIZE_MAIN = None # None for a full run
    run_pipeline(sample_size=SAMPLE_SIZE_MAIN)