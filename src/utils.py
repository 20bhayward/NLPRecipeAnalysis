# src/utils.py

import ast
import logging
import re
import spacy
import time
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set, Union

from .config import RECIPE_STOP_WORDS # Import custom stop words

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variable for SpaCy NLP model and combined stop words ---
NLP = None
ALL_STOP_WORDS = None

def _load_spacy_model():
    """Loads the spaCy model ('en_core_web_sm') once and sets combined stop words."""
    global NLP, ALL_STOP_WORDS
    if NLP is None: # Load only if not already loaded or previously failed
        try:
            logging.info("Loading spaCy model 'en_core_web_sm'...")
            # Disable components not needed for lemmatization/tokenization to speed up loading/processing
            NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logging.info("spaCy model loaded successfully.")
            # Combine spaCy's default stop words with custom recipe-specific ones
            ALL_STOP_WORDS = SPACY_STOP_WORDS.union(RECIPE_STOP_WORDS)
            # Optimize stop word checking by setting the 'is_stop' flag directly in the vocab
            for word in ALL_STOP_WORDS:
                NLP.vocab[word].is_stop = True
            logging.info(f"Combined spaCy and custom stop words. Total unique: {len(ALL_STOP_WORDS)}.")
        except OSError:
            logging.error("SpaCy model 'en_core_web_sm' not found. Download it: python -m spacy download en_core_web_sm")
            logging.error("Traditional NLP preprocessing (TF-IDF/LDA) requires SpaCy and will be skipped.")
            NLP = "Failed" # Mark as failed to prevent repeated attempts
            ALL_STOP_WORDS = RECIPE_STOP_WORDS # Fallback to custom words only
        except Exception as e:
             logging.error(f"Unexpected error loading spaCy model: {e}", exc_info=True)
             NLP = "Failed"
             ALL_STOP_WORDS = RECIPE_STOP_WORDS
    # Ensure stop words are set even if model loaded previously but somehow ALL_STOP_WORDS is None
    elif ALL_STOP_WORDS is None and NLP != "Failed":
        ALL_STOP_WORDS = SPACY_STOP_WORDS.union(RECIPE_STOP_WORDS)
        for word in ALL_STOP_WORDS:
            NLP.vocab[word].is_stop = True

# --- List Parsing ---
def parse_list(x: Union[str, list]) -> Union[list, str]:
    """
    Safely parses a string representation of a list (e.g., "['a', 'b']").
    Returns the original input if it's not a list-like string or if parsing fails.
    """
    if isinstance(x, str):
        # Basic check for list structure before attempting expensive parsing
        if x.startswith('[') and x.endswith(']'):
            try:
                 parsed = ast.literal_eval(x)
                 # Ensure the parsed result is actually a list
                 return parsed if isinstance(parsed, list) else x
            except (ValueError, SyntaxError, TypeError):
                 logging.debug(f"Could not parse string as list via literal_eval: {x[:100]}...")
                 return x # Return original string if parsing fails
        else:
             return x # Not a list-like string
    # If it's already a list (or not a string), return it directly
    return x

# --- Text Joining ---
def safe_join(x: Union[list, str, None]) -> str:
    """
    Safely joins elements of a list into a single space-separated string.
    Handles non-string items and None values within the list.
    Returns an empty string if input is None or not a list/string.
    """
    if isinstance(x, list):
        # Convert all non-None items to string before joining
        return " ".join(str(item) for item in x if item is not None)
    elif isinstance(x, str):
        return x
    return "" # Return empty string for None or other types

# --- Text Preprocessing (SpaCy) ---
def preprocess_text_spacy(text: str, min_token_len: int = 3) -> Tuple[str, List[str]]:
    """
    Applies SpaCy preprocessing: tokenization, lemmatization, stop word & punctuation removal.
    Requires the 'en_core_web_sm' spaCy model to be loaded via _load_spacy_model().

    Args:
        text (str): The input text string.
        min_token_len (int): Minimum length for a token's lemma to be kept.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - The preprocessed text as a single joined string.
            - A list of the processed token lemmas.
            Returns ('', []) if preprocessing fails, input is invalid, or SpaCy model is unavailable.
    """
    _load_spacy_model() # Load the spacy model
    
    # Check if NLP model is available and input is valid
    if NLP is None or NLP == "Failed":
        # Logging handled within _load_spacy_model on failure
        return "", []
    if not isinstance(text, str) or not text.strip():
        return "", [] # Handle non-strings or empty strings

    try:
        # Process text using the loaded SpaCy pipeline
        doc = NLP(text)
        processed_tokens = []

        # Iterate through tokens and apply filters
        for token in doc:
            # Check flags set during loading/processing: stop word, punctuation, space, alpha only
            if (not token.is_stop and
                not token.is_punct and
                not token.is_space and
                token.is_alpha and # Keep only alphabetic tokens
                len(token.lemma_) >= min_token_len):
                processed_tokens.append(token.lemma_.lower()) # Use lowercased lemma

        joined_text = " ".join(processed_tokens)
        return joined_text, processed_tokens

    except Exception as e:
        logging.error(f"Error during spaCy preprocessing for text starting with '{text[:50]}...': {e}", exc_info=True)
        return "", []


# --- Topic Coherence Calculation ---
def calculate_coherence_score(lda_model, top_n_words_list: List[List[str]], token_lists: List[List[str]], coherence_measure: str = 'c_v') -> Optional[float]:
    """
    Calculates the coherence score for topics using Gensim's CoherenceModel.

    Args:
        lda_model: The trained LDA model object (used only for logging context here).
        top_n_words_list (List[List[str]]): List of top words for each topic.
        token_lists (List[List[str]]): List of processed tokens for each document in the corpus.
        coherence_measure (str): The coherence measure ('c_v', 'u_mass', etc.).

    Returns:
        Optional[float]: The calculated coherence score, or None if calculation fails.
    """
    if not token_lists or not any(token_lists):
        logging.error("Cannot calculate coherence: 'token_lists' is empty or contains only empty lists.")
        return None
    if not top_n_words_list or not any(top_n_words_list):
        logging.error("Cannot calculate coherence: 'top_n_words_list' is empty or contains only empty lists.")
        return None

    try:
        logging.debug("Creating Gensim dictionary and corpus for coherence calculation...")
        # Create Gensim dictionary and BoW corpus from the token lists
        dictionary = Dictionary(token_lists)
        corpus = [dictionary.doc2bow(text) for text in token_lists]

        # Validate dictionary and corpus
        if len(dictionary) == 0:
            logging.error("Gensim dictionary is empty. Cannot calculate coherence.")
            return None
        if not corpus or all(not doc for doc in corpus):
             logging.error("Gensim corpus is empty after BoW conversion. Check token lists and dictionary.")
             return None

        logging.info(f"Calculating coherence score ({coherence_measure})...")
        start_time = time.time()

        coherence_model = CoherenceModel(
            topics=top_n_words_list,
            texts=token_lists,
            corpus=corpus,
            dictionary=dictionary,
            coherence=coherence_measure,
            processes=1 # Start with 1 for stability
        )
        coherence_score = coherence_model.get_coherence()

        end_time = time.time()
        logging.info(f"Coherence ({coherence_measure}): {coherence_score:.4f} (Calculation took {end_time - start_time:.2f}s)")

        # Check for non-finite results which can sometimes occur
        if not np.isfinite(coherence_score):
            logging.warning(f"Coherence score calculation resulted in non-finite value: {coherence_score}")
            return None

        return coherence_score

    except ImportError:
        logging.error("Gensim library not found. Please install `gensim` to calculate coherence.")
        return None
    except Exception as e:
        logging.error(f"Error calculating coherence score: {e}", exc_info=True)
        return None

# --- Embedding Helper Functions ---
def find_nearest_neighbors(query_vector: np.ndarray, embedding_matrix: np.ndarray, k: int = 5, metric: str = 'cosine') -> np.ndarray:
    """
    Finds indices of the k nearest neighbors for a query vector within an embedding matrix.

    Args:
        query_vector (np.ndarray): The vector to find neighbors for (1D or 2D with one row).
        embedding_matrix (np.ndarray): Matrix of embeddings to search within (2D).
        k (int): Number of nearest neighbors to return.
        metric (str): Distance metric ('cosine' or 'euclidean').

    Returns:
        np.ndarray: Array of indices of the k nearest neighbors in embedding_matrix.

    Raises:
        ValueError: If metric is unsupported or inputs dimensions mismatch.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1) # Ensure query is 2D
    if query_vector.shape[1] != embedding_matrix.shape[1]:
        raise ValueError("Query vector dimensions must match embedding matrix dimensions.")
    if k <= 0:
        raise ValueError("Number of neighbors 'k' must be positive.")
    if k > embedding_matrix.shape[0]:
        logging.warning(f"k ({k}) > number of samples ({embedding_matrix.shape[0]}). Returning all indices.")
        k = embedding_matrix.shape[0]

    if metric == 'cosine':
        # Cosine similarity: Higher is better. Find highest similarities.
        similarities = cosine_similarity(query_vector, embedding_matrix)[0]
        # Argsort sorts ascending, take the last k indices and reverse for descending order
        nearest_indices = np.argsort(similarities)[-k:][::-1]
    elif metric == 'euclidean':
        # Euclidean distance: Lower is better. Find lowest distances.
        distances = euclidean_distances(query_vector, embedding_matrix)[0]
        # Argsort sorts ascending, take the first k indices
        nearest_indices = np.argsort(distances)[:k]
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    return nearest_indices

def get_representative_texts(cluster_centers: np.ndarray, embeddings: np.ndarray, original_texts: pd.Series, k: int = 5, metric: str = 'cosine') -> Dict[int, List[str]]:
    """
    Finds representative text examples for each cluster center by finding the
    nearest neighbors in the embedding space where clustering was performed.

    Args:
        cluster_centers (np.ndarray): Coordinates of cluster centers (n_clusters, n_features).
        embeddings (np.ndarray): Embedding matrix used for clustering (n_samples, n_features).
        original_texts (pd.Series): Original text data corresponding to embeddings (n_samples).
                                     Index must align with embedding matrix rows.
        k (int): Number of representative texts per cluster.
        metric (str): Metric for finding nearest neighbors ('cosine' or 'euclidean').

    Returns:
        Dict[int, List[str]]: Dictionary mapping cluster index to a list of top k
                              representative text examples. Returns {"error": ...} on major issues.
    """
    # Input validation
    if not isinstance(original_texts, pd.Series):
        logging.error("Input 'original_texts' must be a pandas Series.")
        return {"error": "original_texts must be a pandas Series"}
    if len(original_texts) != embeddings.shape[0]:
        logging.error(f"Length mismatch: original_texts ({len(original_texts)}) vs embeddings rows ({embeddings.shape[0]}).")
        return {"error": "Texts length mismatch with embeddings"}
    if cluster_centers.shape[1] != embeddings.shape[1]:
         logging.error(f"Dimension mismatch: cluster centers ({cluster_centers.shape[1]}) vs embeddings ({embeddings.shape[1]}).")
         return {"error": "Dimension mismatch between centers and embeddings"}

    representative_examples = {}
    num_clusters = cluster_centers.shape[0]
    logging.info(f"Finding {k} representative texts per cluster using '{metric}' metric...")

    for i in range(num_clusters):
        center = cluster_centers[i]
        try:
            # Find indices of texts closest to the current cluster center in the relevant embedding space
            neighbor_indices = find_nearest_neighbors(center, embeddings, k=k, metric=metric)

            # Retrieve corresponding original texts using iloc for position-based lookup
            # find_nearest_neighbors already ensures k <= n_samples
            examples = original_texts.iloc[neighbor_indices].tolist()
            representative_examples[i] = examples

            # Log first few examples (truncated) for debugging
            if examples:
                 log_examples = '; '.join([str(e)[:50]+'...' if len(str(e)) > 50 else str(e) for e in examples[:3]])
                 logging.debug(f"Cluster {i} Rep Texts (Top {min(k, len(examples))} examples): {log_examples}")
            else:
                 logging.debug(f"Cluster {i} Rep Texts: No examples found for indices {neighbor_indices}")

        except IndexError as e:
             logging.error(f"IndexError retrieving texts for cluster {i} (Indices: {neighbor_indices}): {e}", exc_info=True)
             representative_examples[i] = ["Error: IndexError retrieving texts"]
        except Exception as e:
             logging.error(f"Unexpected error retrieving texts for cluster {i}: {e}", exc_info=True)
             representative_examples[i] = ["Error: Failed to retrieve texts"]

    return representative_examples

# --- Stability and Comparison Helpers ---

def calculate_adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculates the Adjusted Rand Index (ARI) to measure clustering similarity."""
    try:
        labels_true = np.asarray(labels_true, dtype=int)
        labels_pred = np.asarray(labels_pred, dtype=int)
        # ARI is undefined for trivial clusterings (all points in one cluster)
        if len(np.unique(labels_true)) <= 1 or len(np.unique(labels_pred)) <= 1:
             logging.warning("Cannot calculate ARI for trivial clusterings (<= 1 cluster). Returning NaN.")
             return np.nan
        return adjusted_rand_score(labels_true, labels_pred)
    except ValueError as e:
        # Catch other potential errors from adjusted_rand_score
        logging.warning(f"Could not calculate ARI: {e}")
        return np.nan

def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculates the Jaccard similarity (Intersection over Union) between two sets."""
    if not isinstance(set1, set) or not isinstance(set2, set):
        logging.warning("Inputs to Jaccard similarity must be sets.")
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    # Avoid division by zero if both sets are empty
    return intersection / union if union > 0 else 0.0

def find_best_config(results_dict: dict, metric_key: str, higher_is_better: bool = True) -> Optional[tuple]:
    """
    Finds the configuration key corresponding to the best metric score in ablation results.

    Args:
        results_dict (dict): Ablation results dictionary. Keys are config tuples,
                             values are dicts containing metrics.
        metric_key (str): Key of the metric to optimize (e.g., 'silhouette_score').
        higher_is_better (bool): True if higher metric score is better.

    Returns:
        Optional[tuple]: Configuration tuple (key) that yielded the best valid score,
                         or None if no valid results found.
    """
    best_score = -np.inf if higher_is_better else np.inf
    best_config = None
    valid_results_found = False

    if not results_dict:
        logging.warning("Cannot find best config: Results dictionary is empty.")
        return None

    for config, results in results_dict.items():
        # Check if the result entry is valid and contains the metric
        if isinstance(results, dict) and metric_key in results:
            score = results[metric_key]

            # Check if the score is a valid, finite numerical type
            if score is not None and isinstance(score, (int, float, np.number)) and np.isfinite(score):
                valid_results_found = True
                if higher_is_better:
                    if score > best_score:
                        best_score = score
                        best_config = config
                else: # Lower is better
                    if score < best_score:
                        best_score = score
                        best_config = config
            else:
                logging.debug(f"Skipping config {config}: Invalid or non-finite score '{score}' for metric '{metric_key}'.")

    if not valid_results_found:
         logging.warning(f"Could not find best config: No valid scores found for metric '{metric_key}'.")
         return None

    logging.info(f"Best configuration found for '{metric_key}': {best_config} (Score: {best_score:.4f})")
    return best_config