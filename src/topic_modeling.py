# src/topic_modeling.py

import pandas as pd
import logging
import time
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import resample # For stability analysis
import numpy as np
from typing import Optional, Tuple, Dict, List, Any

# Use relative import within the package
from .utils import calculate_coherence_score # find_best_config is used in main.py
from .config import (
    N_JOBS, RANDOM_STATE, LDA_TRAD_COHERENCE_MEASURE,
    STABILITY_N_RUNS, STABILITY_SUBSAMPLE_RATIO, OUTLIER_N
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Count Vectorization ---
def vectorize_text_count(texts: List[str], max_features: int, min_df: int = 1, max_df: float = 1.0) -> Tuple[Optional[CountVectorizer], Optional[np.ndarray]]:
    """ Vectorizes text data using CountVectorizer. Assumes stop words already removed. """
    if not texts:
        logging.error("Cannot vectorize: Input text list is empty.")
        return None, None

    logging.info(f"Vectorizing text using CountVectorizer (max_features={max_features}, min_df={min_df}, max_df={max_df})...")
    try:
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=None # Stop words should be handled during preprocessing (e.g., SpaCy)
        )
        count_matrix = vectorizer.fit_transform(texts)
        logging.info(f"Count matrix shape: {count_matrix.shape}")

        if count_matrix.shape[0] == 0 or count_matrix.shape[1] == 0:
            logging.error("Count Vectorization resulted in an empty matrix. Check input texts and parameters (min_df, max_df).")
            return None, None

        # Return dense array for easier use with sklearn LDA, mindful of memory
        return vectorizer, count_matrix.toarray()
    except ValueError as ve:
        logging.error(f"Count Vectorization failed: {ve}. This might happen if vocabulary is empty after filtering.", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during Count Vectorization: {e}", exc_info=True)
        return None, None

# --- LDA Modeling ---
def perform_lda(data_matrix: np.ndarray, n_topics: int, random_state: int, max_iter: int = 10, learning_method: str = 'online') -> Optional[LatentDirichletAllocation]:
    """ Performs Latent Dirichlet Allocation using scikit-learn. """
    if data_matrix is None:
        logging.error("Cannot perform LDA: Input data matrix is None.")
        return None
    if n_topics <= 0:
        logging.error(f"Cannot perform LDA: n_topics must be > 0, got {n_topics}.")
        return None
    if data_matrix.shape[0] == 0:
         logging.error("Cannot perform LDA: Input data matrix has 0 samples.")
         return None
    # Check if n_topics is valid given the number of features (vocabulary size)
    if data_matrix.shape[1] < n_topics:
         logging.error(f"Cannot perform LDA: n_topics ({n_topics}) is greater than the number of features/vocabulary size ({data_matrix.shape[1]}).")
         return None

    logging.info(f"Performing LDA (n_topics={n_topics}, max_iter={max_iter}, learning_method={learning_method})...")
    start_time = time.time()
    try:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method=learning_method, # 'batch' or 'online'
            random_state=random_state,
            n_jobs=N_JOBS # Use configured number of cores
        )
        # Ensure input data is non-negative (should be true for CountVectorizer output)
        if np.any(data_matrix < 0):
             logging.error("Input matrix for LDA contains negative values. Check vectorization.")
             return None

        lda.fit(data_matrix)
        end_time = time.time()
        logging.info(f"LDA training took {end_time - start_time:.2f} seconds.")

        # Basic checks on the trained model
        if not hasattr(lda, 'components_') or lda.components_ is None:
            logging.error("LDA training finished, but model components are missing.")
            return None
        if not np.all(np.isfinite(lda.components_)):
             logging.warning("LDA components contain non-finite values (NaN/inf).")

        return lda

    except Exception as e:
        logging.error(f"LDA training failed: {e}", exc_info=True)
        return None

# --- Topic Interpretation ---
def get_top_words_per_topic(lda_model: LatentDirichletAllocation, vectorizer: CountVectorizer, n_words: int = 10) -> Tuple[Dict[int, List[str]], List[List[str]]]:
    """ Extracts the top N words for each LDA topic based on word weights. """
    topic_words_dict = {}
    top_words_list_for_coherence = [] # Required format for coherence calculation

    if lda_model is None or vectorizer is None:
        logging.error("Cannot get top words: LDA model or vectorizer is missing.")
        return {}, []
    if not hasattr(lda_model, 'components_'):
         logging.error("Cannot get top words: LDA model has no 'components_' attribute.")
         return {}, []

    logging.info(f"Extracting top {n_words} words per topic...")
    try:
        feature_names = vectorizer.get_feature_names_out()
        if feature_names is None or len(feature_names) == 0:
            logging.error("No feature names found in vectorizer.")
            return {}, []

        # lda_model.components_ shape: (n_topics, n_features)
        for topic_idx, topic_distribution in enumerate(lda_model.components_):
            # Get indices of words sorted by weight (descending)
            top_word_indices = topic_distribution.argsort()[:-n_words - 1:-1]
            # Map indices back to words, checking bounds
            valid_indices = [i for i in top_word_indices if i < len(feature_names)]
            top_words = [feature_names[i] for i in valid_indices]

            topic_words_dict[topic_idx] = top_words
            top_words_list_for_coherence.append(top_words)
            logging.debug(f"Topic {topic_idx}: {', '.join(top_words)}")

        return topic_words_dict, top_words_list_for_coherence
    except Exception as e:
        logging.error(f"Error getting top words per topic: {e}", exc_info=True)
        return {}, []

# --- LDA Ablation Study ---
def run_lda_ablation(
    df: pd.DataFrame,
    processed_text_column: str, # Expects column with (joined_text, token_list) tuples
    topic_values: List[int],
    max_features_list: List[int],
    min_df_list: List[int],
    max_df_list: List[float],
    random_state: int
) -> Dict[Tuple, Dict]:
    """
    Runs LDA using CountVectorizer for multiple parameter combinations and evaluates using coherence.

    Args:
        df (pd.DataFrame): Input dataframe.
        processed_text_column (str): Column containing preprocessed (text_string, token_list) tuples.
        topic_values (list): List of number of topics (k) to test.
        max_features_list (list): List of max_features values for CountVectorizer.
        min_df_list (list): List of min_df values for CountVectorizer.
        max_df_list (list): List of max_df values for CountVectorizer.
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[Tuple, Dict]: Results for each parameter combination.
              Keys: (n_topics, max_f, min_df, max_df)
              Values: Dict containing 'coherence_score_cv', 'top_words', etc.
    """
    logging.info("======= Starting LDA Ablation Study (CountVec) =======")
    ablation_results = {}

    # --- 1. Prepare Data ---
    if processed_text_column not in df.columns:
        logging.error(f"Processed text column '{processed_text_column}' not found. Aborting LDA ablation.")
        return {}

    logging.info(f"Using processed text data from column: '{processed_text_column}'")
    try:
        # Extract joined strings for vectorization and token lists for coherence
        processed_data = df[processed_text_column].dropna().tolist()
        # Filter out any entries that are not tuples of (str, list)
        valid_data = [item for item in processed_data if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list)]
        if not valid_data:
             logging.error(f"No valid (string, list) tuples found in column '{processed_text_column}'. Check preprocessing.")
             return {}

        texts_joined = [item[0] for item in valid_data]
        texts_tokenized = [item[1] for item in valid_data]

        # Further filter: ensure non-empty strings and token lists after preprocessing
        valid_indices = [i for i, (s, t) in enumerate(zip(texts_joined, texts_tokenized)) if s.strip() and t]
        if not valid_indices:
            logging.error(f"Preprocessing yielded empty texts or tokens from '{processed_text_column}'. Aborting.")
            return {}

        # Use only valid, aligned texts and tokens
        texts_joined = [texts_joined[i] for i in valid_indices]
        texts_tokenized = [texts_tokenized[i] for i in valid_indices]
        logging.info(f"Number of valid documents for topic modeling: {len(texts_joined)}")

    except Exception as e:
         logging.error(f"Error extracting data from '{processed_text_column}': {e}. Ensure it contains (string, list) tuples.", exc_info=True)
         return {}


    # --- 2. Generate Parameter Combinations ---
    param_combinations = list(itertools.product(
        topic_values, max_features_list, min_df_list, max_df_list
    ))
    total_combinations = len(param_combinations)
    logging.info(f"Total LDA (CountVec) parameter combinations to test: {total_combinations}")

    # --- 3. Run Ablation Loop ---
    for i, params in enumerate(param_combinations):
        n_topics, max_f, min_df, max_df_val = params
        config_key = (n_topics, max_f, min_df, max_df_val)
        logging.info(f"\n--- Running LDA Ablation {i+1}/{total_combinations}: {config_key} ---")

        result_entry = {
            'n_topics': n_topics, 'max_features': max_f, 'min_df': min_df, 'max_df': max_df_val,
            f'coherence_{LDA_TRAD_COHERENCE_MEASURE}': None, 'top_words': {}, 'error': None
        }

        # a. Vectorize Text (CountVectorizer)
        vectorizer, count_matrix = vectorize_text_count(
            texts_joined, max_features=max_f, min_df=min_df, max_df=max_df_val
        )
        if vectorizer is None or count_matrix is None:
            result_entry['error'] = 'Count Vectorization failed'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to vectorization failure.")
            continue

        # b. LDA Modeling
        lda_model = perform_lda(count_matrix, n_topics=n_topics, random_state=random_state)
        if lda_model is None:
            result_entry['error'] = 'LDA failed'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to LDA failure.")
            continue

        # c. Get Top Words per Topic
        top_words_dict, top_words_list = get_top_words_per_topic(lda_model, vectorizer)
        result_entry['top_words'] = top_words_dict
        if not top_words_dict:
             logging.warning(f"Could not extract top words for config {config_key}.")

        # d. Calculate Coherence Score (using original tokenized lists)
        coherence_score = None
        if top_words_list and texts_tokenized:
            coherence_score = calculate_coherence_score(
                lda_model=lda_model, # Passed for context/logging if needed inside function
                top_n_words_list=top_words_list,
                token_lists=texts_tokenized, # Use the preprocessed token lists
                coherence_measure=LDA_TRAD_COHERENCE_MEASURE
            )
            result_entry[f'coherence_{LDA_TRAD_COHERENCE_MEASURE}'] = coherence_score
        else:
            logging.warning(f"Skipping coherence calculation for {config_key} due to missing top words or token lists.")

        # e. Store Results
        ablation_results[config_key] = result_entry
        logging.info(f"Finished config {config_key}. Coherence ({LDA_TRAD_COHERENCE_MEASURE}): {coherence_score if coherence_score is not None else 'N/A'}")

    logging.info("======= Finished LDA Ablation Study (CountVec) =======")

    # --- 4. Print Summary Table ---
    print("\n--- LDA (CountVec) Ablation Summary ---")
    metric_name = f'coherence_{LDA_TRAD_COHERENCE_MEASURE}'
    print(f"Config (n_topics, max_f, min_df, max_df) | Coherence ({LDA_TRAD_COHERENCE_MEASURE}) | Top 3 Words (Topic 0)")
    print("-" * 90)
    # Sort results by coherence score (descending, handle None/NaN)
    sorted_results = sorted(
        [(cfg, res) for cfg, res in ablation_results.items() if isinstance(res, dict) and res.get(metric_name) is not None and np.isfinite(res.get(metric_name))],
        key=lambda item: item[1][metric_name],
        reverse=True
    )
    sorted_results.extend([(cfg, res) for cfg, res in ablation_results.items() if not isinstance(res, dict) or res.get(metric_name) is None or not np.isfinite(res.get(metric_name))])

    for config, result in sorted_results:
        if isinstance(result, dict):
            score = result.get(metric_name, 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (float, np.number)) and np.isfinite(score) else str(score)
            top_words_dict = result.get('top_words', {})
            cluster_0_terms = top_words_dict.get(0, ['N/A'])[:3]
            terms_str = ', '.join(cluster_0_terms)
            error = result.get('error')
            config_str = str(config).ljust(40)

            if error:
                 print(f"{config_str} | Error: {error}")
            else:
                 print(f"{config_str} | {score_str.ljust(19)} | {terms_str}")
        else:
             print(f"{str(config).ljust(40)} | Invalid result entry")

    return ablation_results


# --- Stability and Outlier Analysis ---
def analyze_lda_stability_coherence(
    texts_joined: List[str],
    texts_tokenized: List[List[str]],
    vectorizer_params: Dict[str, Any],
    lda_params: Dict[str, Any],
    n_runs: int = STABILITY_N_RUNS,
    subsample_ratio: float = STABILITY_SUBSAMPLE_RATIO,
    random_state: int = RANDOM_STATE
) -> Tuple[Optional[float], Optional[float]]:
    """
    Analyzes LDA stability by calculating topic coherence (e.g., c_v) on multiple subsamples.

    Args:
        texts_joined (List[str]): List of joined preprocessed text strings.
        texts_tokenized (List[List[str]]): List of corresponding token lists for coherence calculation.
        vectorizer_params (Dict[str, Any]): Parameters for CountVectorizer.
        lda_params (Dict[str, Any]): Parameters for LDA (n_topics).
        n_runs (int): Number of subsampling runs.
        subsample_ratio (float): Proportion of data for each run.
        random_state (int): Base random state for reproducibility.

    Returns:
        Tuple[Optional[float], Optional[float]]: Mean and standard deviation of the
                                                 coherence scores across runs, or (None, None).
    """
    if not texts_joined or not texts_tokenized or len(texts_joined) != len(texts_tokenized):
        logging.error("Invalid input texts for LDA stability analysis.")
        return None, None

    logging.info(f"--- Starting LDA Stability Analysis (Coherence {LDA_TRAD_COHERENCE_MEASURE}, runs={n_runs}, ratio={subsample_ratio}) ---")
    start_time = time.time()
    coherence_scores = []
    n_samples = len(texts_joined)
    subsample_size = int(n_samples * subsample_ratio)

    if subsample_size < lda_params.get('n_topics', 1):
        logging.error(f"Subsample size ({subsample_size}) may be too small for LDA with {lda_params.get('n_topics', 'N/A')} topics.")

    for i in range(n_runs):
        run_seed = random_state + i # Vary seed for resampling and LDA initialization
        logging.debug(f"Running stability iteration {i+1}/{n_runs}...")
        try:
            # Create subsample
            subsample_indices = resample(np.arange(n_samples), n_samples=subsample_size, replace=False, random_state=run_seed)
            sub_texts_joined = [texts_joined[j] for j in subsample_indices]
            sub_texts_tokenized = [texts_tokenized[j] for j in subsample_indices] # Keep tokens aligned

            # 1. Vectorize subsample
            vectorizer_run, count_matrix_run = vectorize_text_count(sub_texts_joined, **vectorizer_params)
            if vectorizer_run is None or count_matrix_run is None:
                 logging.warning(f"Vectorization failed on stability run {i+1}. Skipping.")
                 continue

            # 2. Train LDA on subsample
            lda_run = perform_lda(count_matrix_run, random_state=run_seed, **lda_params)
            if lda_run is None:
                 logging.warning(f"LDA failed on stability run {i+1}. Skipping.")
                 continue

            # 3. Get top words from the subsample model
            _, top_words_list_run = get_top_words_per_topic(lda_run, vectorizer_run)
            if not top_words_list_run:
                 logging.warning(f"Could not get top words on stability run {i+1}. Skipping coherence.")
                 continue

            # 4. Calculate coherence using the subsample's tokenized texts
            coherence_run = calculate_coherence_score(
                lda_model=lda_run,
                top_n_words_list=top_words_list_run,
                token_lists=sub_texts_tokenized, # Use the token lists corresponding to the subsample
                coherence_measure=LDA_TRAD_COHERENCE_MEASURE
            )

            if coherence_run is not None and np.isfinite(coherence_run):
                coherence_scores.append(coherence_run)
                logging.debug(f"  Run {i+1} Coherence ({LDA_TRAD_COHERENCE_MEASURE}): {coherence_run:.4f}")
            else:
                logging.warning(f"Coherence calculation failed or yielded non-finite score on run {i+1}.")

        except Exception as e:
            logging.warning(f"Error during LDA stability run {i+1}: {e}. Skipping.", exc_info=True)
            continue

    if not coherence_scores:
        logging.error("LDA stability analysis failed: No valid coherence scores obtained.")
        mean_coherence, std_coherence = None, None
    else:
        mean_coherence = np.mean(coherence_scores)
        std_coherence = np.std(coherence_scores)
        logging.info(f"LDA Stability Analysis Result: Mean Coherence ({LDA_TRAD_COHERENCE_MEASURE}) = {mean_coherence:.4f}, Std Dev = {std_coherence:.4f} (based on {len(coherence_scores)} runs)")

    logging.info(f"--- Finished LDA Stability Analysis (Took {time.time() - start_time:.2f}s) ---")
    return mean_coherence, std_coherence


def find_lda_outliers(doc_topic_matrix: np.ndarray, n_outliers: int = OUTLIER_N) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Identifies outlier documents based on low maximum topic probability assignment.
    Outliers are documents the LDA model is least certain about.

    Args:
        doc_topic_matrix (np.ndarray): Document-topic probability matrix (n_docs, n_topics).
        n_outliers (int): Number of outlier documents to identify.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            - Indices of the top N outlier documents (sorted by lowest max probability).
            - Corresponding maximum topic probabilities for these outliers.
            Returns (None, None) if calculation fails.
    """
    if doc_topic_matrix is None or doc_topic_matrix.ndim != 2 or doc_topic_matrix.shape[0] == 0:
        logging.warning("Skipping LDA outlier analysis: Invalid document-topic matrix.")
        return None, None
    if n_outliers <= 0:
        logging.warning("Skipping LDA outlier analysis: n_outliers must be positive.")
        return None, None

    logging.info(f"--- Starting LDA Outlier Analysis (Top {n_outliers} outliers based on lowest max probability) ---")
    start_time = time.time()

    try:
        # Find the maximum probability for each document across all topics
        max_topic_probs = np.max(doc_topic_matrix, axis=1)

        # Handle potential non-finite probabilities
        if not np.all(np.isfinite(max_topic_probs)):
             logging.warning("Document-topic matrix contains non-finite probabilities. Filtering them.")
             valid_indices_probs = np.where(np.isfinite(max_topic_probs))[0]
             if len(valid_indices_probs) == 0:
                 logging.error("No finite max topic probabilities found.")
                 return None, None
             probs_to_sort = max_topic_probs[valid_indices_probs]
             original_indices_map = valid_indices_probs # Map sorted indices back to original
        else:
             probs_to_sort = max_topic_probs
             original_indices_map = np.arange(len(max_topic_probs)) # Direct mapping

        # Argsort sorts ascending: first N indices correspond to lowest max probabilities
        sorted_prob_indices = np.argsort(probs_to_sort)
        # Select the indices corresponding to the lowest N probabilities
        outlier_indices_in_sorted = sorted_prob_indices[:min(n_outliers, len(probs_to_sort))]

        # Map back to original document indices and get their probabilities
        outlier_original_indices = original_indices_map[outlier_indices_in_sorted]
        outlier_max_probs = max_topic_probs[outlier_original_indices]

        logging.info(f"Identified {len(outlier_original_indices)} potential outlier documents (low topic certainty).")
        logging.info(f"--- Finished LDA Outlier Analysis (Took {time.time() - start_time:.2f}s) ---")
        return outlier_original_indices, outlier_max_probs

    except Exception as e:
        logging.error(f"Error during LDA outlier analysis: {e}", exc_info=True)
        return None, None