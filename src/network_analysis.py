# src/network_analysis.py
import pandas as pd
import networkx as nx
import community as community_louvain # python-louvain library
from collections import Counter
import logging
import time
from typing import Optional, Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_ingredients(df: pd.DataFrame, ingredient_col: str = 'ingredients', sample_size: Optional[int] = None, random_state: Optional[int] = None) -> Optional[List[List[str]]]:
    """
    Samples the DataFrame, extracts, cleans (lowercase, strip), and validates ingredient lists.

    Args:
        df (pd.DataFrame): The input DataFrame.
        ingredient_col (str): Name of the column containing ingredient lists (expects list format).
        sample_size (Optional[int]): Number of recipes to sample. If None, use all.
        random_state (Optional[int]): Random state for sampling reproducibility.

    Returns:
        Optional[List[List[str]]]: A list of lists, where each inner list contains
                                    cleaned ingredient strings for a recipe. Returns
                                    None if the ingredient column is missing or no valid lists found.
    """
    if ingredient_col not in df.columns:
        logging.error(f"Ingredient column '{ingredient_col}' not found in DataFrame.")
        return None

    # Apply sampling if requested
    df_processed = df
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        logging.info(f"Sampling {sample_size} recipes for network analysis (random_state={random_state}).")
        try:
            df_processed = df.sample(n=sample_size, random_state=random_state)
        except ValueError as e:
            logging.error(f"Sampling failed ({e}). Using full DataFrame. Check if sample_size <= DataFrame length.")
            df_processed = df # Fallback to full df if sampling fails
    elif sample_size is not None and sample_size >= len(df):
        logging.info("Sample size >= DataFrame length. Using full DataFrame for network analysis.")
        df_processed = df
    else:
        logging.info("Using full DataFrame for network analysis.")


    logging.info("Preparing ingredient lists (ensuring list format, lowercasing, stripping)...")
    start_time = time.time()
    ingredients_list = []
    invalid_entries = 0
    # Iterate through the specified column, ensuring entries are lists
    for item in df_processed[ingredient_col].dropna():
        if isinstance(item, list):
            # Clean ingredients within the list (convert to lower string, strip whitespace)
            cleaned_ings = [str(ing).lower().strip() for ing in item if str(ing).strip()]
            if cleaned_ings: # Keep only lists that are non-empty after cleaning
                ingredients_list.append(cleaned_ings)
            else:
                invalid_entries += 1
        else:
            logging.debug(f"Skipping non-list entry in ingredient column: {type(item)} - {str(item)[:50]}...")
            invalid_entries += 1

    if invalid_entries > 0:
        logging.warning(f"Skipped {invalid_entries} invalid or empty ingredient entries.")

    if not ingredients_list:
        logging.error("No valid ingredient lists found after preparation.")
        return None

    logging.info(f"Prepared {len(ingredients_list)} ingredient lists in {time.time() - start_time:.2f} seconds.")
    return ingredients_list

def filter_ingredients_by_freq(ingredients_list: List[List[str]], min_freq: int, max_freq: int) -> Tuple[List[List[str]], Counter]:
    """ Filters ingredients based on their frequency across all provided recipe lists. """
    logging.info(f"Filtering ingredients by frequency (min={min_freq}, max={max_freq})...")
    start_time = time.time()

    # Calculate frequency of all ingredients
    flat_ingredients = [ing for sublist in ingredients_list for ing in sublist]
    if not flat_ingredients:
        logging.warning("No ingredients found to filter.")
        return [], Counter()
    freq_counts = Counter(flat_ingredients)

    # Identify ingredients within the desired frequency range
    ingredients_to_keep = {
        ing for ing, count in freq_counts.items() if min_freq <= count <= max_freq
    }

    if not ingredients_to_keep:
        logging.warning(f"No ingredients remain after frequency filtering (min={min_freq}, max={max_freq}). Check thresholds.")
        return [], freq_counts # Return original counts for context

    # Re-filter the original list structure, keeping only ingredients in the allowed set
    filtered_ingredients_list = [
        [ing for ing in sublist if ing in ingredients_to_keep]
        for sublist in ingredients_list
    ]
    # Remove recipes that become empty after filtering
    filtered_ingredients_list = [sublist for sublist in filtered_ingredients_list if sublist]

    num_original_unique = len(freq_counts)
    num_filtered_unique = len(ingredients_to_keep)
    logging.info(f"Filtered ingredients: {num_filtered_unique} unique ingredients remaining out of {num_original_unique}.")
    logging.info(f"Number of recipes remaining after filtering: {len(filtered_ingredients_list)}.")
    logging.info(f"Frequency filtering took {time.time() - start_time:.2f} seconds.")

    return filtered_ingredients_list, freq_counts # Return original frequencies

def calculate_cooccurrence(filtered_ingredients_list: List[List[str]]) -> Counter:
    """ Calculates pairwise co-occurrence counts for ingredients within each recipe list. """
    if not filtered_ingredients_list:
        logging.warning("Cannot calculate co-occurrence: Input ingredient list is empty.")
        return Counter()

    logging.info("Calculating ingredient co-occurrences...")
    start_time = time.time()
    co_occurrence = Counter()

    for ing_list in filtered_ingredients_list:
        # Use unique ingredients within each recipe to count pairs only once per recipe
        unique_ings_in_recipe = set(ing_list)
        # Generate sorted pairs only if there are >= 2 unique ingredients
        if len(unique_ings_in_recipe) > 1:
            sorted_ings = sorted(list(unique_ings_in_recipe))
            for i in range(len(sorted_ings)):
                for j in range(i + 1, len(sorted_ings)):
                    pair = (sorted_ings[i], sorted_ings[j]) # Canonical pair (ing1, ing2) where ing1 < ing2
                    co_occurrence[pair] += 1

    end_time = time.time()
    logging.info(f"Co-occurrence calculation took {end_time - start_time:.2f} seconds.")
    logging.info(f"Found {len(co_occurrence)} unique co-occurring pairs (before thresholding).")
    return co_occurrence

def build_ingredient_network(co_occurrence_counts: Counter, threshold: int = 1) -> nx.Graph:
    """ Builds a NetworkX graph from co-occurrence counts, adding edges above a threshold. """
    logging.info(f"Building ingredient network graph (co-occurrence threshold >= {threshold})...")
    start_time = time.time()
    G = nx.Graph()
    edges_added = 0
    nodes_added = set() # Track nodes added via edges

    if not co_occurrence_counts:
        logging.warning("No co-occurrence counts provided. Returning empty graph.")
        return G

    # Add edges (and corresponding nodes) where co-occurrence count meets the threshold
    for (ing1, ing2), count in co_occurrence_counts.items():
        if count >= threshold:
            G.add_edge(ing1, ing2, weight=count)
            nodes_added.add(ing1)
            nodes_added.add(ing2)
            edges_added += 1

    logging.info(f"Network built with {len(nodes_added)} nodes and {edges_added} edges "
                 f"(after applying threshold {threshold}).")
    logging.info(f"Network building took {time.time() - start_time:.2f} seconds.")
    # Node attributes like frequency can be added later if needed (done in pipeline func)
    return G

def detect_communities(graph: nx.Graph, random_state: Optional[int] = None) -> Tuple[Optional[Dict[Any, int]], int]:
    """
    Detects communities using the Louvain algorithm. Handles disconnected graphs.

    Args:
        graph (nx.Graph): The input graph.
        random_state (Optional[int]): Random state for the Louvain algorithm reproducibility.

    Returns:
        Tuple[Optional[Dict[Any, int]], int]:
            - Dictionary mapping node to community ID. Returns None if detection fails.
            - Number of communities detected. Returns 0 if detection fails.
    """
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        logging.warning("Graph is empty or invalid. Cannot detect communities.")
        return None, 0

    logging.info("Detecting communities using Louvain algorithm...")
    start_time = time.time()

    # Check graph connectivity for awareness, although Louvain handles disconnected graphs.
    if not nx.is_connected(graph):
        num_components = nx.number_connected_components(graph)
        logging.warning(f"Graph is not connected ({num_components} components). Louvain will run on the full graph.")

    try:
        # Use edge weights for community detection if they exist
        partition = community_louvain.best_partition(graph, weight='weight', random_state=random_state)
        end_time = time.time()
        logging.info(f"Community detection took {end_time - start_time:.2f} seconds.")

        if not partition:
             logging.warning("Louvain algorithm returned an empty partition.")
             return None, 0

        num_communities = len(set(partition.values()))
        logging.info(f"Detected {num_communities} communities.")

        return partition, num_communities

    except Exception as e:
        logging.error(f"Louvain community detection failed: {e}", exc_info=True)
        return None, 0


def run_network_analysis_pipeline(
    df: pd.DataFrame,
    ingredient_col: str,
    sample_size: Optional[int],
    min_freq: int,
    max_freq: int,
    co_occurrence_threshold: int,
    random_state: int
) -> Optional[Dict[str, Any]]:
    """
    Runs the full ingredient network analysis pipeline.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ingredient_col (str): Name of the ingredient list column.
        sample_size (Optional[int]): Number of recipes to sample for analysis.
        min_freq (int): Minimum frequency for an ingredient node.
        max_freq (int): Maximum frequency for an ingredient node.
        co_occurrence_threshold (int): Minimum co-occurrence count for an edge.
        random_state (int): Random state for sampling and community detection.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing results:
            'graph': NetworkX graph object.
            'partition': Node to community ID mapping.
            'num_communities': Number of detected communities.
            'ingredient_frequencies': Counter of original frequencies (before filtering).
            'graph_summary': Basic stats about the final graph.
            'error': Error message if a step failed critically.
            Returns None only if ingredient preparation fails completely.
    """
    logging.info("======= Starting Ingredient Network Analysis =======")
    pipeline_start = time.time()
    results = {'error': None}

    # 1. Prepare Ingredients (Sample, Clean)
    ingredients_list = prepare_ingredients(df, ingredient_col, sample_size, random_state)
    if ingredients_list is None:
        logging.error("Failed to prepare ingredients. Aborting network analysis.")
        results['error'] = "Ingredient preparation failed"
        return None # Critical failure

    # 2. Filter by Frequency
    filtered_list, freq_counts = filter_ingredients_by_freq(ingredients_list, min_freq, max_freq)
    results['ingredient_frequencies'] = freq_counts # Store original frequencies for context
    if not filtered_list:
         logging.error("No ingredients left after frequency filtering. Aborting network analysis.")
         results['error'] = "Frequency filtering removed all ingredients"
         return results

    # 3. Calculate Co-occurrence
    co_occur = calculate_cooccurrence(filtered_list)
    if not co_occur:
        logging.warning("No co-occurrences found after filtering. Network will be empty.")

    # 4. Build Network Graph
    graph = build_ingredient_network(co_occur, threshold=co_occurrence_threshold)
    results['graph'] = graph
    # Add frequency as a node attribute using the original frequencies
    nodes_in_graph = list(graph.nodes())
    nodes_added_count = 0
    for node in nodes_in_graph:
        frequency = freq_counts.get(node, 0) # Get original frequency
        graph.nodes[node]['frequency'] = frequency
        nodes_added_count += 1
    if nodes_added_count > 0:
        logging.info(f"Added frequency attribute to {nodes_added_count} nodes in the graph.")

    # Add basic graph summary to results
    graph_summary = {}
    if graph is not None:
        graph_summary['num_nodes'] = graph.number_of_nodes()
        graph_summary['num_edges'] = graph.number_of_edges()
        if graph.number_of_nodes() > 0:
            try:
                graph_summary['density'] = nx.density(graph)
                graph_summary['num_connected_components'] = nx.number_connected_components(graph)
            except Exception:
                graph_summary['density'] = 'N/A'
                graph_summary['num_connected_components'] = 'N/A'
        else:
             graph_summary['density'] = 0.0
             graph_summary['num_connected_components'] = 0
    results['graph_summary'] = graph_summary
    logging.info(f"Graph Summary: {graph_summary}")

    # 5. Detect Communities
    if graph is not None and graph.number_of_nodes() > 0:
        partition, num_communities = detect_communities(graph, random_state=random_state)
        results['partition'] = partition
        results['num_communities'] = num_communities
        if partition is None:
             logging.warning("Community detection failed.")
             results['error'] = results.get('error') or "Community detection failed" # Keep previous error if any
    else:
        logging.warning("Skipping community detection as graph is empty.")
        results['partition'] = {}
        results['num_communities'] = 0

    logging.info(f"======= Finished Ingredient Network Analysis (Total time: {time.time() - pipeline_start:.2f}s) =======")
    return results