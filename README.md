
# Recipe NLP Analysis Pipeline

This project analyzes over two million recipes from the RecipeNLG dataset using various Natural Language Processing (NLP) techniques. It extracts features, performs clustering (TF-IDF/SVD and Transformer Embeddings), topic modeling (LDA), ingredient network analysis, stability checks, and outlier detection to gain insights into recipe structures and themes.

## Features

*   **Data Loading & Cleaning:** Loads the RecipeNLG dataset and parses list-like columns (ingredients, directions).
*   **Feature Engineering:** Creates features like `num_ingredients`, `num_steps`, and `title_length`.
*   **Text Preprocessing:** Utilizes SpaCy for lemmatization, stop word removal, and filtering of ingredient and direction text.
*   **TF-IDF/SVD Clustering:** Performs K-Means clustering on TF-IDF vectors of ingredients, optionally reduced by Truncated SVD. Includes an ablation study over hyperparameters.
*   **Transformer Embeddings:** Generates sentence embeddings for ingredients and directions using pre-trained models (e.g., DistilBERT) via the `transformers` library. Supports GPU acceleration (CUDA/MPS).
*   **Embedding Clustering:** Performs K-Means clustering on ingredient embeddings, optionally reduced by UMAP. Includes an ablation study over hyperparameters.
*   **LDA Topic Modeling:** Applies Latent Dirichlet Allocation (LDA) on Count Vectors of directions to identify latent topics. Includes an ablation study and coherence evaluation.
*   **Ingredient Network Analysis:** Builds a co-occurrence network of ingredients, detects communities using the Louvain algorithm, and analyzes network properties.
*   **Stability Analysis:** Assesses the robustness of the best clustering (KMeans ARI) and topic modeling (LDA Coherence) models using subsampling.
*   **Outlier Detection:** Identifies recipes that are outliers based on distance from cluster centroids (KMeans) or low topic probability assignment (LDA).
*   **Modular Pipeline:** Orchestrated via `src/main.py` with control flags to enable/disable specific steps.
*   **Visualization:** A separate Jupyter Notebook (`notebooks/visualizations.ipynb`) loads saved results and generates plots for EDA, cluster analysis (UMAP/SVD), topic word clouds, network communities, and outlier highlighting.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/20bhayward/NLPRecipeAnalysis.git
    cd NLPRecipeAnalysis
    ```

2.  **Setup Python Environment:**
    It's highly recommended to use a virtual environment. Requires Python 3.8 or newer.
    ```bash
    # Create a virtual environment named 'venv'
    python -m venv venv

    # Activate the environment
    # On Windows (Git Bash or WSL):
    source venv/bin/activate
    # On Windows (Command Prompt/PowerShell):
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    First, ensure you have a `requirements.txt` file. You can generate one using `pip freeze > requirements.txt` after installing the necessary libraries, or create it manually listing the core dependencies identified in the code (pandas, numpy, scikit-learn, spacy, transformers, torch, gensim, networkx, python-louvain, umap-learn, matplotlib, seaborn, tqdm, wordcloud).

    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `torch` might require specific commands depending on your system and whether you want CPU or GPU (CUDA/MPS) support. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).*

4.  **Download Dataset:**
    *   Download the `full_dataset.csv` from the [RecipeNLG Kaggle Dataset](https://www.kaggle.com/datasets/saldenisov/recipenlg).
    *   Place the downloaded `full_dataset.csv` file inside the `data/` directory.

5.  **Download SpaCy Model:**
    The pipeline uses the `en_core_web_sm` model for text preprocessing. Download it using:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Running the Analysis Pipeline

The main analysis pipeline is executed via `src/main.py`.

1.  **Configure Pipeline Steps:**
    *   Open `src/main.py`.
    *   Modify the `RUN_...` boolean flags near the top of the file to enable or disable specific pipeline stages (e.g., `RUN_TRANSFORMER_EMBEDDINGS = False` to skip generating embeddings if they already exist).

2.  **Configure Sample Size:**
    *   In `src/main.py`, find the `if __name__ == "__main__":` block at the bottom.
    *   Set the `SAMPLE_SIZE_MAIN` variable to the desired number of recipes to process (e.g., `50000`) or set it to `None` to use the full dataset. Using a smaller sample size is recommended for initial runs and development.

3.  **Run the Script:**
    Navigate to the project's root directory in your terminal (where the `src` folder is located) and run:
    ```bash
    python src/main.py
    ```
    *   The pipeline will execute the enabled steps. Progress and information will be printed to the console and saved to `results/pipeline_run.log`.
    *   Running the full pipeline, especially embedding generation and ablation studies on the full dataset, can take a significant amount of time and computational resources (GPU recommended for embeddings).

## Visualizing Results

After the pipeline (`main.py`) has successfully generated the output files in the `results/` directory, you can visualize the findings using the Jupyter Notebook.

1.  **Start Jupyter:**
    Ensure your virtual environment is activated. Navigate to the project's root directory and start Jupyter Lab or Notebook:
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```

2.  **Open and Run the Notebook:**
    *   In the Jupyter interface, navigate into the `notebooks/` directory.
    *   Open `visualizations.ipynb`.
    *   Run the cells sequentially. The notebook will load the data and results from the `results/` directory and generate various plots and summaries.

## Configuration Details

*   **Paths & Columns:** Core file paths and DataFrame column names are defined in `src/config.py`. Modify these if your setup differs.
*   **Hyperparameters:** Parameters for ablation studies (KMeans K values, TF-IDF/LDA features/df, UMAP settings) are defined in `src/config.py`. You can adjust these lists to explore different configurations.
*   **Models:** The transformer model (`TRANSFORMER_MODEL`) and pooling strategies are set in `src/config.py`.
*   **Pipeline Flow:** Enable/disable major analysis steps using the `RUN_...` flags in `src/main.py`.
*   **Sampling:** Control the dataset size used for analysis via `SAMPLE_SIZE_MAIN` in `src/main.py`.

## Output Files (`results/` directory)

*   `processed_recipes_sample_with_text.pkl`: A pandas DataFrame containing the sampled (or full) dataset after cleaning, feature engineering, and potentially adding processed text columns and cluster labels.
*   `all_analysis_results.pkl`: A Python dictionary saved using pickle. Contains nested dictionaries holding the results from ablation studies (parameters, scores), stability analysis (ARI, coherence stats), and outlier analysis (indices, scores, texts).
*   `ingredient_embeddings.npy`: NumPy array containing the generated transformer embeddings for ingredients.
*   `direction_embeddings.npy`: NumPy array containing the generated transformer embeddings for directions.
*   `network_analysis_results.pkl`: A Python dictionary containing the network summary (nodes, edges, etc.) and the community partition mapping (node -> community ID), excluding the potentially large NetworkX graph object itself.
*   `pipeline_run.log`: A text file logging the execution steps, warnings, and errors from the `main.py` run.