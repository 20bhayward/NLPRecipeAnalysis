# src/clustering.py

import pandas as pd
import logging
import time
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.utils import resample # For stability analysis subsampling
import numpy as np
import umap # umap-learn package
from typing import Optional, Tuple, Dict, List, Any

# Use relative import within the package
from .utils import (
    get_representative_texts,
    calculate_adjusted_rand_index,
    find_best_config
)
from .config import N_JOBS, RANDOM_STATE, STABILITY_N_RUNS, STABILITY_SUBSAMPLE_RATIO, OUTLIER_N

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TF-IDF Vectorization ---
def vectorize_text_tfidf(texts: pd.Series, max_features: int, min_df: int = 1, max_df: float = 1.0) -> Tuple[Optional[TfidfVectorizer], Optional[np.ndarray]]:
    """Vectorizes text data using TF-IDF."""
    if texts.empty:
        logging.error("Cannot vectorize: Input text series is empty.")
        return None, None
    logging.info(f"Vectorizing text using TF-IDF (max_features={max_features}, min_df={min_df}, max_df={max_df})...")
    # Assuming stop words removed during SpaCy preprocessing
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=None
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        # Check if vectorization produced usable output
        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            logging.error("TF-IDF Vectorization resulted in an empty matrix. Check input texts and parameters (min_df, max_df).")
            return None, None
        if not np.all(np.isfinite(tfidf_matrix.data)):
             logging.warning("TF-IDF matrix contains non-finite values.")
        # Convert sparse to dense for downstream use (consider memory for very large matrices)
        return vectorizer, tfidf_matrix.toarray()
    except ValueError as ve:
         logging.error(f"TF-IDF Vectorization failed: {ve}. This might happen if vocabulary is empty after filtering.", exc_info=True)
         return None, None
    except Exception as e:
         logging.error(f"Unexpected error during TF-IDF Vectorization: {e}", exc_info=True)
         return None, None


# --- Dimensionality Reduction (SVD for TF-IDF) ---
def reduce_dimensions_svd(matrix: np.ndarray, n_components: int, random_state: int) -> Tuple[Optional[TruncatedSVD], Optional[np.ndarray]]:
    """Reduces dimensionality using Truncated SVD."""
    if matrix is None or matrix.shape[0] == 0:
        logging.error("Cannot reduce dimensions: Input matrix is None or empty.")
        return None, None
    # Ensure n_components is valid
    if n_components <= 0:
         logging.error(f"Invalid n_components for SVD: {n_components}. Must be > 0.")
         return None, None
    if n_components >= matrix.shape[1]:
         logging.warning(f"n_components ({n_components}) >= number of features ({matrix.shape[1]}). Skipping SVD.")
         # Return None for svd_model, but the original matrix as the "embedding"
         return None, matrix
    if n_components >= matrix.shape[0]:
        logging.warning(f"n_components ({n_components}) >= number of samples ({matrix.shape[0]}). Skipping SVD.")
        return None, matrix


    logging.info(f"Reducing dimensions using Truncated SVD (n_components={n_components})...")
    start_time = time.time()
    try:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        embedding = svd.fit_transform(matrix)
        end_time = time.time()
        logging.info(f"SVD embedding shape: {embedding.shape} (Took {end_time - start_time:.2f}s)")
        if hasattr(svd, 'explained_variance_ratio_'):
            explained_variance = svd.explained_variance_ratio_.sum()
            logging.info(f"SVD explained variance ratio (top {n_components}): {explained_variance:.4f}")
        return svd, embedding
    except Exception as e:
        logging.error(f"Truncated SVD failed: {e}", exc_info=True)
        return None, None


# --- Dimensionality Reduction (UMAP for Embeddings) ---
def reduce_dimensions_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, n_components: int, metric: str = 'cosine', random_state: Optional[int] = None) -> Tuple[Optional[umap.UMAP], Optional[np.ndarray]]:
    """Reduces dimensionality using UMAP."""
    if embeddings is None or embeddings.shape[0] < n_neighbors:
         logging.error(f"Cannot perform UMAP: Input embeddings invalid or n_neighbors ({n_neighbors}) > n_samples ({embeddings.shape[0] if embeddings is not None else 0}).")
         return None, None
    if n_components <= 0:
        logging.error(f"Invalid n_components for UMAP: {n_components}. Must be > 0.")
        return None, None

    logging.info(f"Reducing dimensions using UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, metric={metric})...")
    start_time = time.time()
    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            n_jobs=N_JOBS, # Use configured number of cores
            verbose=False
        )
        umap_embedding = reducer.fit_transform(embeddings)
        end_time = time.time()
        logging.info(f"UMAP embedding shape: {umap_embedding.shape} (Took {end_time - start_time:.2f}s)")
        if not np.all(np.isfinite(umap_embedding)):
             logging.warning("UMAP embedding contains non-finite values.")
        return reducer, umap_embedding
    except Exception as e:
        logging.error(f"UMAP dimension reduction failed: {e}", exc_info=True)
        return None, None

# --- K-Means Clustering ---
def perform_kmeans(data_matrix: np.ndarray, n_clusters: int, random_state: int) -> Tuple[Optional[KMeans], Optional[np.ndarray]]:
    """Performs K-Means clustering."""
    if data_matrix is None or data_matrix.shape[0] < n_clusters:
        logging.error(f"Cannot perform KMeans: Invalid data or n_clusters ({n_clusters}) > n_samples ({data_matrix.shape[0] if data_matrix is not None else 0}).")
        return None, None
    if n_clusters <= 0:
        logging.error(f"Cannot perform KMeans: n_clusters must be > 0, got {n_clusters}.")
        return None, None


    logging.info(f"Performing KMeans clustering (k={n_clusters})...")
    start_time = time.time()
    try:
        # n_init='auto' is recommended and default in newer sklearn
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', max_iter=300)
        # Ensure data is finite before passing to KMeans
        if not np.all(np.isfinite(data_matrix)):
            logging.error("Input data for KMeans contains non-finite values. Imputing or cleaning needed.")
            return None, None

        clusters = kmeans.fit_predict(data_matrix)
        end_time = time.time()
        logging.info(f"KMeans clustering took {end_time - start_time:.2f} seconds.")
        if not hasattr(kmeans, 'cluster_centers_') or kmeans.cluster_centers_ is None:
             logging.error("KMeans finished but cluster centers are not available.")
             return None, None
        if not np.all(np.isfinite(kmeans.cluster_centers_)):
             logging.warning("KMeans cluster centers contain non-finite values.")

        return kmeans, clusters
    except Exception as e:
        logging.error(f"KMeans clustering failed: {e}", exc_info=True)
        return None, None

# --- Clustering Evaluation ---
def evaluate_clustering_silhouette(data_matrix: np.ndarray, clusters: np.ndarray, metric: str = 'cosine', sample_size: int = 5000, random_state: Optional[int] = None) -> Optional[float]:
    """Calculates the Silhouette Score for clustering evaluation."""
    if data_matrix is None or clusters is None:
        logging.warning("Skipping Silhouette Score: Invalid input data or clusters.")
        return None

    n_labels = len(set(clusters))
    n_samples = data_matrix.shape[0]

    # Silhouette score requires 2 <= n_labels <= n_samples - 1
    if not (2 <= n_labels <= n_samples - 1):
        logging.warning(f"Skipping Silhouette Score: Invalid number of labels ({n_labels}) for {n_samples} samples.")
        return None

    logging.info(f"Calculating Silhouette Score (metric={metric})...")
    start_time = time.time()
    try:
        # Use sampling for large datasets to speed up calculation
        if n_samples > sample_size:
             logging.debug(f"Using random sample of size {sample_size} for Silhouette calculation.")
             indices = np.random.choice(n_samples, sample_size, replace=False)
             score = silhouette_score(data_matrix[indices], clusters[indices], metric=metric, random_state=random_state)
        else:
             score = silhouette_score(data_matrix, clusters, metric=metric)

        logging.info(f"Silhouette Score ({metric}): {score:.4f}")
        end_time = time.time()
        logging.debug(f"Silhouette calculation took {end_time - start_time:.2f} seconds.")
        if not np.isfinite(score):
             logging.warning(f"Silhouette Score calculation resulted in non-finite value: {score}")
             return None
        return score
    except ValueError as ve:
         logging.error(f"Silhouette Score calculation failed: {ve}. Check data matrix, clusters, and metric ('{metric}').", exc_info=True)
         return None
    except Exception as e:
         logging.error(f"Silhouette Score failed unexpectedly: {e}", exc_info=True)
         return None


# --- Top Terms for TF-IDF Clusters ---
def get_top_terms_per_cluster(kmeans_model: KMeans, vectorizer: TfidfVectorizer, svd_model: Optional[TruncatedSVD] = None, n_terms: int = 10) -> Dict[int, List[str]]:
    """
    Extracts the top terms for each cluster based on TF-IDF centroids.
    Handles optional SVD reduction by inverse transforming centroids.
    """
    cluster_terms = {}
    if kmeans_model is None or vectorizer is None:
        logging.error("Cannot get top terms: KMeans model or vectorizer is missing.")
        return {}

    logging.info("Extracting top terms per cluster (TF-IDF based)...")
    try:
        terms = vectorizer.get_feature_names_out()
        if terms is None or len(terms) == 0:
            logging.error("No feature names found in TF-IDF vectorizer.")
            return {}

        cluster_centers = kmeans_model.cluster_centers_
        if cluster_centers is None:
            logging.error("KMeans model has no cluster centers.")
            return {}

        # If SVD was used, inverse transform centers back to original TF-IDF space
        if svd_model is not None and hasattr(svd_model, 'inverse_transform'):
            try:
                original_space_centroids = svd_model.inverse_transform(cluster_centers)
                logging.debug("Inverse transformed SVD centroids to TF-IDF space.")
            except Exception as e:
                logging.error(f"Failed to inverse transform SVD centers: {e}. Cannot get top terms.", exc_info=True)
                return {}
        # If no SVD model was provided, assume KMeans was run directly on TF-IDF matrix
        elif svd_model is None:
            if cluster_centers.shape[1] != len(terms):
                 logging.error(f"Centroid dimension ({cluster_centers.shape[1]}) mismatch with vocabulary size ({len(terms)}) and no SVD model provided.")
                 return {}
            original_space_centroids = cluster_centers
            logging.debug("Using direct KMeans centroids in TF-IDF space.")
        else:
             logging.error("SVD model provided but cannot inverse transform. Cannot get top terms.")
             return {}


        # Get top terms for each centroid
        order_centroids = original_space_centroids.argsort()[:, ::-1] # Sort term weights descending

        for i in range(order_centroids.shape[0]): # Iterate through clusters
            # Get the indices of the top N terms for this cluster
            term_indices = order_centroids[i, :n_terms]
            # Map indices back to words, ensuring indices are valid
            valid_indices = [ind for ind in term_indices if ind < len(terms)]
            top_cluster_terms = [terms[ind] for ind in valid_indices]

            cluster_terms[i] = top_cluster_terms
            logging.debug(f"Cluster {i} Top Terms: {', '.join(top_cluster_terms)}")

        return cluster_terms
    except Exception as e:
        logging.error(f"Error getting top TF-IDF terms per cluster: {e}", exc_info=True)
        return {}

# --- Ablation Study Functions ---

def run_kmeans_ablation_tfidf(
    df: pd.DataFrame,
    text_column: str,
    k_values: List[int],
    max_features_list: List[int],
    min_df_list: List[int],
    max_df_list: List[float],
    n_components_list: List[int],
    random_state: int
) -> Dict[Tuple, Dict]:
    """
    Runs K-Means clustering with TF-IDF/SVD for multiple parameter combinations.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column with preprocessed text strings.
        k_values (List[int]): List of 'k' (number of clusters) values to test.
        max_features_list (List[int]): List of max_features for TF-IDF.
        min_df_list (List[int]): List of min_df for TF-IDF.
        max_df_list (List[float]): List of max_df for TF-IDF.
        n_components_list (List[int]): List of n_components for SVD reduction.
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[Tuple, Dict]: Dictionary containing results for each parameter combination.
              Keys are tuples: (k, max_f, min_df, max_df, n_comp)
              Values are dicts containing 'silhouette_score', 'top_terms', etc.
    """
    logging.info("======= Starting K-Means Ablation Study (TF-IDF/SVD) =======")
    ablation_results = {}

    # --- 1. Validate Input ---
    if text_column not in df.columns:
        logging.error(f"Text column '{text_column}' not found in DataFrame. Aborting K-Means TF-IDF ablation.")
        return {}
    texts = df[text_column].fillna('').astype(str)
    if texts.empty or texts.str.strip().eq('').all():
         logging.error(f"Text column '{text_column}' is empty or contains only whitespace after cleaning. Aborting.")
         return {}
    logging.info(f"Using text data from column '{text_column}' ({len(texts)} documents).")

    # --- 2. Generate Parameter Combinations ---
    param_combinations = list(itertools.product(
        k_values, max_features_list, min_df_list, max_df_list, n_components_list))
    total_combinations = len(param_combinations)
    logging.info(f"Total K-Means (TF-IDF/SVD) parameter combinations to test: {total_combinations}")

    # --- 3. Run Ablation Loop ---
    for i, params in enumerate(param_combinations):
        k, max_f, min_df, max_df_val, n_comp = params
        config_key = (k, max_f, min_df, max_df_val, n_comp)
        logging.info(f"\n--- Running TF-IDF/SVD Ablation {i+1}/{total_combinations}: {config_key} ---")

        result_entry = {
            'k': k, 'max_features': max_f, 'min_df': min_df, 'max_df': max_df_val, 'n_components': n_comp,
            'silhouette_score': None, 'top_terms': {}, 'error': None
        }

        # Step a: Vectorize Text (TF-IDF)
        vectorizer, tfidf_matrix = vectorize_text_tfidf(texts, max_features=max_f, min_df=min_df, max_df=max_df_val)
        if vectorizer is None or tfidf_matrix is None:
            result_entry['error'] = 'TF-IDF Vectorization failed'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to vectorization failure.")
            continue

        # Step b: Reduce Dimensions (SVD)
        # Note: reduce_dimensions_svd handles n_comp >= features internally by skipping SVD
        svd_model, svd_embedding = reduce_dimensions_svd(tfidf_matrix, n_components=n_comp, random_state=random_state)
        # If SVD failed critically (svd_embedding is None but svd_model is not None), abort for this config.
        if svd_embedding is None and svd_model is not None:
            result_entry['error'] = 'SVD reduction failed critically'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to SVD failure.")
            continue
        # Determine the data to use for KMeans: SVD result if available, otherwise original TF-IDF matrix.
        data_for_kmeans = svd_embedding if svd_embedding is not None else tfidf_matrix
        if data_for_kmeans is None or data_for_kmeans.shape[0] < k:
             result_entry['error'] = 'Data for KMeans is invalid or too few samples'
             ablation_results[config_key] = result_entry
             logging.warning(f"Skipping config {config_key} due to invalid data for KMeans.")
             continue

        # Step c: Perform KMeans
        kmeans_model, clusters = perform_kmeans(data_for_kmeans, n_clusters=k, random_state=random_state)
        if kmeans_model is None or clusters is None:
            result_entry['error'] = 'KMeans failed'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to KMeans failure.")
            continue

        # Step d: Evaluate Clustering (Silhouette Score)
        # Use cosine distance for high-dimensional sparse spaces like TF-IDF/SVD
        silhouette_avg = evaluate_clustering_silhouette(data_for_kmeans, clusters, metric='cosine', random_state=random_state)
        result_entry['silhouette_score'] = silhouette_avg

        # Step e: Get Top Terms
        top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, svd_model=svd_model)
        result_entry['top_terms'] = top_terms

        # Store results for this configuration
        ablation_results[config_key] = result_entry
        logging.info(f"Finished config {config_key}. Silhouette (cosine): {silhouette_avg if silhouette_avg is not None else 'N/A'}")

    logging.info("======= Finished K-Means Ablation Study (TF-IDF/SVD) =======")

    # --- 4. Print Summary Table ---
    print("\n--- K-Means (TF-IDF/SVD) Ablation Summary ---")
    print("Config (k, max_f, min_df, max_df, n_comp) | Silhouette (cosine) | Top 3 Terms (Cluster 0)")
    print("-" * 90)
    # Sort results by silhouette score (descending, handle None)
    sorted_results = sorted(
        [(cfg, res) for cfg, res in ablation_results.items() if isinstance(res, dict) and res.get('silhouette_score') is not None],
        key=lambda item: item[1]['silhouette_score'],
        reverse=True
    )
    sorted_results.extend([(cfg, res) for cfg, res in ablation_results.items() if not isinstance(res, dict) or res.get('silhouette_score') is None])

    for config, result in sorted_results:
        if isinstance(result, dict):
            score = result.get('silhouette_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (float, np.number)) and np.isfinite(score) else str(score)
            cluster_0_terms = result.get('top_terms', {}).get(0, ['N/A'])[:3] # Get top 3 for cluster 0
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


def run_kmeans_ablation_embeddings(
    embeddings: np.ndarray,
    original_texts: pd.Series,
    k_values: List[int],
    use_umap: bool,
    umap_neighbors_list: List[int],
    umap_min_dist_list: List[float],
    umap_components_list: List[int],
    umap_metric: str,
    random_state: int
) -> Dict[Tuple, Dict]:
    """
    Runs K-Means clustering on embeddings, optionally with UMAP, for multiple parameters.

    Args:
        embeddings (np.ndarray): The pre-computed text embeddings.
        original_texts (pd.Series): Original texts corresponding to embeddings (for representative text finding).
        k_values (List[int]): List of 'k' (number of clusters) values to test.
        use_umap (bool): Whether to apply UMAP before KMeans.
        umap_neighbors_list (List[int]): List of n_neighbors for UMAP. Ignored if use_umap is False.
        umap_min_dist_list (List[float]): List of min_dist for UMAP. Ignored if use_umap is False.
        umap_components_list (List[int]): List of n_components for UMAP. Ignored if use_umap is False.
        umap_metric (str): Distance metric for UMAP. Ignored if use_umap is False.
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[Tuple, Dict]: Dictionary containing results for each parameter combination.
              Keys are tuples: (k, umap_n, umap_d, umap_c) if UMAP used, or (k,) otherwise.
              Values are dicts containing 'silhouette_score', 'representative_texts', etc.
    """
    logging.info("======= Starting K-Means Ablation Study (Embeddings) =======")
    ablation_results = {}

    # --- 1. Validate Input ---
    if embeddings is None or len(embeddings) == 0:
        logging.error("Embeddings are missing or empty. Aborting K-Means (Embeddings) ablation.")
        return {}
    if len(original_texts) != embeddings.shape[0]:
        logging.error("Embeddings length does not match original texts length. Aborting.")
        return {}
    if not np.all(np.isfinite(embeddings)):
        logging.warning("Input embeddings contain non-finite values. KMeans or UMAP might fail.")

    # --- 2. Generate Parameter Combinations ---
    if use_umap:
        param_combinations = list(itertools.product(
            k_values, umap_neighbors_list, umap_min_dist_list, umap_components_list))
        logging.info("Using UMAP before K-Means.")
        config_prefix = "UMAP+KMeans"
    else:
        param_combinations = list(itertools.product(k_values))
        logging.info("Running K-Means directly on full embeddings.")
        config_prefix = "KMeans"

    total_combinations = len(param_combinations)
    logging.info(f"Total {config_prefix} parameter combinations to test: {total_combinations}")

    # --- 3. Run Ablation Loop ---
    for i, params in enumerate(param_combinations):
        data_for_kmeans = None
        config_key = params
        umap_params = {}

        logging.info(f"\n--- Running {config_prefix} Ablation {i+1}/{total_combinations}: {config_key} ---")

        result_entry = {
            'k': params[0],
            'use_umap': use_umap,
            'silhouette_score': None,
            'silhouette_metric': None,
            'representative_texts': {},
            'error': None
        }
        if use_umap:
             result_entry.update({'umap_n_neighbors': params[1], 'umap_min_dist': params[2],
                                 'umap_n_components': params[3], 'umap_metric': umap_metric})
             umap_params = {'n_neighbors': params[1], 'min_dist': params[2], 'n_components': params[3], 'metric': umap_metric}


        # Step a: Reduce Dimensions (UMAP), if specified
        if use_umap:
            k, umap_n, umap_d, umap_c = params
            umap_model, umap_embedding = reduce_dimensions_umap(
                embeddings, n_neighbors=umap_n, min_dist=umap_d, n_components=umap_c,
                metric=umap_metric, random_state=random_state)
            if umap_model is None or umap_embedding is None:
                result_entry['error'] = 'UMAP failed'
                ablation_results[config_key] = result_entry
                logging.warning(f"Skipping config {config_key} due to UMAP failure.")
                continue
            data_for_kmeans = umap_embedding
            # Euclidean distance often works well in UMAP's lower-dimensional projected space
            silhouette_metric = 'euclidean'
            rep_text_comparison_metric = 'euclidean' # Use same metric for rep texts in this space
        else:
            k = params[0]
            data_for_kmeans = embeddings
            # Cosine distance is often preferred for high-dimensional raw embeddings
            silhouette_metric = 'cosine'
            rep_text_comparison_metric = 'cosine'

        result_entry['silhouette_metric'] = silhouette_metric

        # Step b: Perform KMeans
        kmeans_model, clusters = perform_kmeans(data_for_kmeans, n_clusters=k, random_state=random_state)
        if kmeans_model is None or clusters is None:
            result_entry['error'] = 'KMeans failed'
            ablation_results[config_key] = result_entry
            logging.warning(f"Skipping config {config_key} due to KMeans failure.")
            continue

        # Step c: Evaluate Clustering (Silhouette Score)
        silhouette_avg = evaluate_clustering_silhouette(data_for_kmeans, clusters, metric=silhouette_metric, random_state=random_state)
        result_entry['silhouette_score'] = silhouette_avg

        # Step d: Get Representative Texts
        # Find representative texts based on proximity to centroids in the space where KMeans was performed.
        try:
            cluster_centers = kmeans_model.cluster_centers_
            logging.debug(f"Finding representative texts in the '{rep_text_comparison_metric}' space (dim={data_for_kmeans.shape[1]}) used for clustering.")
            rep_texts = get_representative_texts(
                 cluster_centers=cluster_centers,
                 embeddings=data_for_kmeans, # Use the same data matrix KMeans ran on
                 original_texts=original_texts,
                 k=5, # Number of representatives per cluster
                 metric=rep_text_comparison_metric # Use the metric appropriate for this space
            )
            result_entry['representative_texts'] = rep_texts
        except Exception as e_rep:
             logging.error(f"Failed to get representative texts for config {config_key}: {e_rep}", exc_info=True)
             result_entry['representative_texts'] = {"error": "Failed to get representative texts"}

        # Store results
        ablation_results[config_key] = result_entry
        logging.info(f"Finished config {config_key}. Silhouette ({silhouette_metric}): {silhouette_avg if silhouette_avg is not None else 'N/A'}")

    logging.info("======= Finished K-Means Ablation Study (Embeddings) =======")

    # --- 4. Print Summary Table ---
    print("\n--- K-Means (Embeddings) Ablation Summary ---")
    if use_umap:
        print("Config (k, umap_n, umap_d, umap_c)       | Silhouette Score | Metric    | Top Rep Text (Cluster 0)")
    else:
        print("Config (k,)                            | Silhouette Score | Metric    | Top Rep Text (Cluster 0)")
    print("-" * 95)
    # Sort results by silhouette score (descending, handle None)
    sorted_results = sorted(
        [(cfg, res) for cfg, res in ablation_results.items() if isinstance(res, dict) and res.get('silhouette_score') is not None],
        key=lambda item: item[1]['silhouette_score'],
        reverse=True
    )
    sorted_results.extend([(cfg, res) for cfg, res in ablation_results.items() if not isinstance(res, dict) or res.get('silhouette_score') is None])

    for config, result in sorted_results:
         if isinstance(result, dict):
            score = result.get('silhouette_score', 'N/A')
            metric = result.get('silhouette_metric', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (float, np.number)) and np.isfinite(score) else str(score)
            rep_texts_dict = result.get('representative_texts', {})
            cluster_0_texts = rep_texts_dict.get(0, ['N/A'])[:1]
            text_str = 'N/A'
            if cluster_0_texts and isinstance(cluster_0_texts[0], str):
                 text_str = cluster_0_texts[0]
                 if len(text_str) > 40: text_str = text_str[:37] + '...'

            error = result.get('error')
            config_str = str(config).ljust(38 if use_umap else 38)
            if error:
                print(f"{config_str} | Error: {error}")
            else:
                print(f"{config_str} | {score_str.ljust(16)} | {metric.ljust(9)} | {text_str}")
         else:
             print(f"{str(config).ljust(38 if use_umap else 38)} | Invalid result entry")

    return ablation_results


# --- Stability and Outlier Analysis ---

def analyze_kmeans_stability(
    data_matrix: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_runs: int = STABILITY_N_RUNS,
    subsample_ratio: float = STABILITY_SUBSAMPLE_RATIO
) -> Optional[float]:
    """
    Analyzes the stability of KMeans clustering using subsampling and Adjusted Rand Index (ARI).

    Args:
        data_matrix (np.ndarray): The data used for clustering.
        n_clusters (int): The number of clusters (k).
        random_state (int): Base random state for reproducibility.
        n_runs (int): Number of subsampling runs.
        subsample_ratio (float): Proportion of data to use in each run.

    Returns:
        Optional[float]: The average ARI score across pairs of runs, or None if analysis fails.
    """
    if data_matrix is None or n_clusters <= 1 or n_runs < 2:
        logging.warning("Skipping KMeans stability analysis due to invalid inputs.")
        return None

    logging.info(f"--- Starting KMeans Stability Analysis (k={n_clusters}, runs={n_runs}, ratio={subsample_ratio}) ---")
    start_time = time.time()
    all_full_labels = []
    n_samples = data_matrix.shape[0]
    subsample_size = int(n_samples * subsample_ratio)

    if subsample_size < n_clusters:
         logging.error(f"Subsample size ({subsample_size}) is less than n_clusters ({n_clusters}). Cannot perform stability analysis.")
         return None

    for i in range(n_runs):
        run_seed = random_state + i # Vary seed for each run's subsampling & KMeans
        try:
            # Create subsample indices
            subsample_indices = resample(np.arange(n_samples), n_samples=subsample_size, replace=False, random_state=run_seed)
            subsample_data = data_matrix[subsample_indices]

            # Perform KMeans on the subsample
            kmeans_run, _ = perform_kmeans(subsample_data, n_clusters=n_clusters, random_state=run_seed)
            if kmeans_run is None:
                logging.warning(f"KMeans failed on stability run {i+1}. Skipping this run.")
                continue

            # Predict labels for the *entire* dataset using the model trained on the subsample
            full_labels_run = kmeans_run.predict(data_matrix)
            all_full_labels.append(full_labels_run)

        except Exception as e:
            logging.warning(f"Error during KMeans stability run {i+1}: {e}. Skipping run.", exc_info=True)
            continue

    if len(all_full_labels) < 2:
        logging.error("KMeans stability analysis failed: Fewer than 2 successful runs completed.")
        return None

    # Calculate pairwise ARI scores
    ari_scores = []
    for i in range(len(all_full_labels)):
        for j in range(i + 1, len(all_full_labels)):
            ari = calculate_adjusted_rand_index(all_full_labels[i], all_full_labels[j])
            # Only include finite ARI scores (can be NaN for trivial clusterings)
            if ari is not None and np.isfinite(ari):
                 ari_scores.append(ari)

    if not ari_scores:
        logging.error("KMeans stability analysis failed: Could not calculate any valid ARI scores.")
        avg_ari = None
    else:
        avg_ari = np.mean(ari_scores)
        logging.info(f"KMeans Stability Analysis Result: Average ARI = {avg_ari:.4f} (based on {len(ari_scores)} pairwise comparisons)")

    logging.info(f"--- Finished KMeans Stability Analysis (Took {time.time() - start_time:.2f}s) ---")
    return avg_ari


def find_kmeans_outliers(data_matrix: np.ndarray, kmeans_model: KMeans, n_outliers: int = OUTLIER_N, metric: str = 'euclidean') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Identifies outlier points in KMeans clustering based on distance to their assigned centroid.

    Args:
        data_matrix (np.ndarray): The data that was clustered.
        kmeans_model (KMeans): The trained KMeans model.
        n_outliers (int): The number of outliers to identify.
        metric (str): Metric to calculate distance ('euclidean' or 'cosine').

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            - Indices of the top N outlier points.
            - The corresponding distances of these outliers to their centroids.
            Returns (None, None) if calculation fails.
    """
    if data_matrix is None or kmeans_model is None or not hasattr(kmeans_model, 'cluster_centers_') or not hasattr(kmeans_model, 'labels_'):
        logging.warning("Skipping KMeans outlier analysis: Invalid input data or model.")
        return None, None

    logging.info(f"--- Starting KMeans Outlier Analysis (Top {n_outliers} outliers, metric={metric}) ---")
    start_time = time.time()

    try:
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        distances = np.full(data_matrix.shape[0], np.nan) # Initialize distances

        # Calculate distance for each point to its assigned centroid based on the metric
        if metric == 'euclidean':
             for i in range(data_matrix.shape[0]):
                 assigned_center = centers[labels[i]]
                 distances[i] = np.linalg.norm(data_matrix[i] - assigned_center)
        elif metric == 'cosine':
             for i in range(data_matrix.shape[0]):
                 assigned_center = centers[labels[i]].reshape(1, -1)
                 point = data_matrix[i].reshape(1,-1)
                 similarity = cosine_similarity(point, assigned_center)[0][0]
                 distances[i] = 1.0 - similarity # Cosine distance = 1 - similarity
        else:
            logging.error(f"Unsupported metric '{metric}' for KMeans outlier distance.")
            return None, None

        if np.isnan(distances).all():
             logging.error("Failed to calculate distances for outlier analysis.")
             return None, None

        # Find indices of points with the largest distances
        # Filter out NaNs before sorting
        valid_indices = np.where(np.isfinite(distances))[0]
        if len(valid_indices) == 0:
            logging.error("No finite distances found for outlier analysis.")
            return None, None

        # Sort only the valid distances/indices in descending order
        sorted_indices_valid = np.argsort(distances[valid_indices])[::-1]
        # Get the original indices corresponding to the top N largest distances
        outlier_indices_valid = valid_indices[sorted_indices_valid[:min(n_outliers, len(valid_indices))]]
        outlier_distances = distances[outlier_indices_valid]

        logging.info(f"Identified {len(outlier_indices_valid)} potential outliers.")
        logging.info(f"--- Finished KMeans Outlier Analysis (Took {time.time() - start_time:.2f}s) ---")
        return outlier_indices_valid, outlier_distances

    except Exception as e:
        logging.error(f"Error during KMeans outlier analysis: {e}", exc_info=True)
        return None, None