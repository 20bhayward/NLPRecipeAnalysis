# src/embedding_generation.py

import logging
import time
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # Progress bar
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    """Checks for CUDA GPU, Apple MPS, or falls back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
             logging.warning(f"Could not get GPU name, but CUDA is available: {e}")
             logging.info("Using GPU.")
    elif torch.backends.mps.is_available(): # Check for Apple Silicon GPU
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders (MPS) on GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for embeddings generation.")
    return device

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs mean pooling on token embeddings, respecting the attention mask.

    Args:
        model_output: Output from the transformer model (contains `last_hidden_state`).
        attention_mask (torch.Tensor): Attention mask for input tokens.

    Returns:
        torch.Tensor: Mean-pooled sentence embedding.
    """
    token_embeddings = model_output[0] # First element usually has token embeddings
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum embeddings where attention mask is 1 (ignore padding)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum mask to get count of actual tokens per sequence
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Clamp for stability
    return sum_embeddings / sum_mask

def cls_pooling(model_output) -> torch.Tensor:
    """
    Extracts the embedding of the [CLS] token (first token).

    Args:
        model_output: Output from the transformer model.

    Returns:
        torch.Tensor: The [CLS] token embedding.
    """
    return model_output[0][:, 0] # [CLS] token is the first one

def generate_transformer_embeddings(
    texts: pd.Series,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    pooling_strategy: str = 'mean'
) -> Optional[np.ndarray]:
    """
    Generates embeddings for a Series of texts using a pre-trained transformer model.

    Args:
        texts (pd.Series): Text data to embed.
        model_name (str): Hugging Face model name/path.
        batch_size (int): Number of texts to process per batch.
        max_length (int): Max sequence length for tokenizer (truncates longer texts).
        pooling_strategy (str): Method to aggregate token embeddings ('mean' or 'cls').

    Returns:
        Optional[np.ndarray]: NumPy array of embeddings (n_texts, embedding_dim),
                              or None if a critical error occurs. May return fewer rows
                              than input texts if batches fail.
    """
    logging.info(f"Starting transformer embedding generation using model: {model_name}")
    start_time = time.time()

    if not isinstance(texts, pd.Series):
        logging.error("Input 'texts' must be a pandas Series.")
        return None
    if texts.empty:
        logging.warning("Input 'texts' Series is empty. Returning None.")
        return None

    # Ensure texts are strings and handle NaNs
    texts_list = texts.fillna('').astype(str).tolist()
    n_texts = len(texts_list)
    if n_texts == 0:
        logging.warning("Input 'texts' Series contains no valid strings after cleaning. Returning None.")
        return None
    logging.info(f"Preparing to process {n_texts} texts...")

    device = get_device()

    try:
        logging.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        model.to(device) # Move model to selected device
        model.eval() # Set to evaluation mode
        logging.info(f"Model '{model_name}' loaded successfully onto {device}.")
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer '{model_name}': {e}", exc_info=True)
        return None

    all_embeddings = []
    batches_processed = 0
    batches_failed = 0
    with torch.no_grad(): # Disable gradients for inference
        for i in tqdm(range(0, n_texts, batch_size), desc=f"Generating Embeddings ({pooling_strategy} pooling)"):
            batch_texts = texts_list[i : min(i + batch_size, n_texts)]
            if not batch_texts: continue

            try:
                inputs = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,       # Pad to max length in batch
                    truncation=True,    # Truncate longer sequences
                    return_tensors='pt' # Return PyTorch tensors
                )
            except Exception as e:
                 logging.error(f"Error during tokenization for batch starting at index {i}: {e}", exc_info=True)
                 batches_failed += 1
                 continue # Skip failed batch

            try:
                 inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                 logging.error(f"Error moving batch tensors to device {device} for batch {i}: {e}")
                 batches_failed += 1
                 continue

            try:
                outputs = model(**inputs)
            except Exception as e:
                 logging.error(f"Error during model inference on batch starting at index {i}: {e}", exc_info=True)
                 batches_failed += 1
                 continue

            # Apply pooling strategy
            try:
                if pooling_strategy == 'mean':
                    batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
                elif pooling_strategy == 'cls':
                    batch_embeddings = cls_pooling(outputs)
                else:
                    logging.error(f"Invalid pooling strategy: '{pooling_strategy}'. Aborting.")
                    return None # Critical error if pooling is wrong
            except Exception as e:
                logging.error(f"Error during pooling for batch starting at index {i}: {e}", exc_info=True)
                batches_failed += 1
                continue

            # Move embeddings to CPU (if generated on GPU/MPS) and store as NumPy array
            all_embeddings.append(batch_embeddings.cpu().numpy())
            batches_processed += 1

    if batches_failed > 0:
         logging.warning(f"{batches_failed} batch(es) failed during embedding generation.")

    if not all_embeddings:
        logging.error("No embeddings were generated. Check input data and logs for errors.")
        return None

    # Concatenate embeddings from all successful batches
    try:
        final_embeddings = np.vstack(all_embeddings)
    except Exception as e:
        logging.error(f"Failed to stack embeddings from batches: {e}", exc_info=True)
        return None

    end_time = time.time()
    logging.info(f"Finished generating embeddings. Success/Fail Batches: {batches_processed}/{batches_failed}. Total Embeddings: {final_embeddings.shape[0]}.")
    logging.info(f"Final embedding matrix shape: {final_embeddings.shape}. Time taken: {end_time - start_time:.2f} seconds.")

    # Check for NaN/inf values in the final matrix
    if np.isnan(final_embeddings).any() or np.isinf(final_embeddings).any():
        nan_count = np.isnan(final_embeddings).sum()
        inf_count = np.isinf(final_embeddings).sum()
        logging.warning(f"Embeddings contain {nan_count} NaN or {inf_count} Inf values. This might indicate issues.")

    # Final check: does the number of embeddings match the number of input texts?
    if final_embeddings.shape[0] != n_texts:
         logging.warning(f"Number of generated embeddings ({final_embeddings.shape[0]}) does not match the number of input texts ({n_texts}) due to batch failures. Returning partial results.")

    return final_embeddings