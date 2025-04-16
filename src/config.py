# src/config.py

import os

# --- Directory Paths ---
_SRC_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')

# --- File Paths ---
RECIPE_FILE = os.path.join(DATA_DIR, "full_dataset.csv")
NETWORK_RESULTS_FILE = os.path.join(RESULTS_DIR, 'network_analysis_results.pkl')
PROCESSED_DF_FILE = os.path.join(RESULTS_DIR, 'processed_recipes_sample_with_text.pkl')
ALL_RESULTS_FILE = os.path.join(RESULTS_DIR, 'all_analysis_results.pkl')
INGREDIENT_EMB_FILE = os.path.join(RESULTS_DIR, 'ingredient_embeddings.npy')
DIRECTION_EMB_FILE = os.path.join(RESULTS_DIR, 'direction_embeddings.npy')

# --- Core DataFrame Column Names ---
TITLE_COLUMN = 'title'
INGREDIENTS_LIST_COLUMN = 'ingredients' # Expects Python list format after cleaning
DIRECTIONS_LIST_COLUMN = 'directions' # Expects Python list format after cleaning

# --- Derived Column Names ---
# SpaCy processed text columns for traditional methods
CLUSTER_TEXT_COLUMN_SPACY = 'ingredients_processed_spacy' # Holds tuple (joined_text, token_list)
CLUSTER_TEXT_COLUMN_SPACY_STR = CLUSTER_TEXT_COLUMN_SPACY + '_str' # Holds only joined_text string
TOPIC_TEXT_COLUMN_SPACY = 'directions_processed_spacy' # Holds tuple (joined_text, token_list)
# Raw joined text columns for transformer inputs
INGREDIENTS_RAW_JOINED_COL = 'ingredients_joined_raw'
DIRECTIONS_RAW_JOINED_COL = 'directions_joined_raw'

# --- Clustering Ablation Parameters (TF-IDF/SVD on SpaCy Ingredients) ---
KMEANS_TRAD_K_VALUES = [5, 8, 12]
KMEANS_TRAD_MAX_FEATURES = [1000, 1500]
KMEANS_TRAD_MIN_DF = [3, 5]
KMEANS_TRAD_MAX_DF = [0.85, 0.95]
KMEANS_TRAD_N_COMPONENTS_SVD = [50, 100]

# --- Topic Modeling Ablation Parameters (CountVec/LDA on SpaCy Directions) ---
LDA_TRAD_N_TOPICS = [5, 8, 12]
LDA_TRAD_MAX_FEATURES = [1000, 1500]
LDA_TRAD_MIN_DF = [5, 10]
LDA_TRAD_MAX_DF = [0.85, 0.90]
LDA_TRAD_COHERENCE_MEASURE = 'c_v' # Gensim coherence measure to use

# --- Transformer Embedding Parameters ---
TRANSFORMER_MODEL = "distilbert-base-uncased" # Example: "sentence-transformers/all-MiniLM-L6-v2"
INGREDIENTS_EMB_POOLING = 'mean' # Pooling strategy: 'mean' or 'cls'
DIRECTIONS_EMB_POOLING = 'mean'
INGREDIENTS_EMB_MAX_LEN = 64 # Max sequence length for tokenizer
DIRECTIONS_EMB_MAX_LEN = 256
EMBEDDING_BATCH_SIZE = 64 # Adjust based on GPU memory

# --- Clustering Ablation Parameters (K-Means on Transformer Embeddings - Ingredients) ---
KMEANS_EMB_K_VALUES = [5, 8, 12]
# UMAP parameters (Set USE_UMAP=False to run KMeans directly on embeddings)
KMEANS_EMB_USE_UMAP = True
KMEANS_EMB_UMAP_NEIGHBORS = [15, 30] # UMAP n_neighbors
KMEANS_EMB_UMAP_MIN_DIST = [0.1, 0.3] # UMAP min_dist
KMEANS_EMB_UMAP_COMPONENTS = [5, 10] # UMAP target dimensions for clustering
KMEANS_EMB_UMAP_METRIC = 'cosine' # UMAP distance metric (applied to input embeddings)

# --- Network Analysis Parameters (Ingredients) ---
INGREDIENT_NETWORK_SAMPLE_SIZE = 5000 # Number of recipes to sample for network building
INGREDIENT_MIN_FREQ = 5 # Min frequency for an ingredient to be included
INGREDIENT_MAX_FREQ = 30000 # Max frequency for an ingredient to be included (removes extremely common items)
CO_OCCURRENCE_THRESHOLD = 6 # Min times two ingredients must appear together to form an edge

# --- Stability & Outlier Analysis Parameters ---
STABILITY_N_RUNS = 5 #Number of subsampling runs for stability analysis
STABILITY_SUBSAMPLE_RATIO = 0.8 # Proportion of data for each stability run
OUTLIER_N = 10 # Target number of outliers to identify per model/method

# --- General Configuration ---
RANDOM_STATE = 42 # Seed for reproducibility
N_JOBS = -1 # Use all available CPU cores for parallelizable tasks (-1)

# --- Stop Words ---
# Custom stop words specific to recipes, combined with spaCy's defaults in utils.py
RECIPE_STOP_WORDS = {
    "add", "mix", "stir", "combine", "blend", "heat", "cook", "bake",
    "preheat", "pour", "place", "set", "aside", "serve", "remove", "cover",
    "cup", "cups", "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g",
    "milliliter", "milliliters", "ml", "liter", "liters", "l",
    "inch", "inches", "cm", "mm",
    "minute", "minutes", "min", "hour", "hours", "hr", "hrs",
    "degree", "degrees", "fahrenheit", "celsius", "f", "c",
    "oven", "pan", "bowl", "skillet", "pot", "saucepan", "dish", "plate", "sheet", "baking",
    "make", "sure", "use", "well", "approximately", "optional", "large", "small", "medium",
    "package", "pkg", "can", "cans", "jar", "jars", "container", "bag", "box",
    "ingredient", "ingredients", "direction", "directions", "recipe", "step", "steps",
    "water", "oil", "salt", "pepper", "sugar", "flour", "butter", "egg", "eggs",
    "get", "need", "want", "like", "also", "just", "really", "much", "until", "slice", "cut",
    "room", "temperature", "melt", "dissolve", "drain", "bring", "boil", "reduce", "simmer",
    "cool", "beat", "whisk", "fold", "sprinkle", "garnish", "chop", "dice", "mince", "peel",
    "taste", "season", "adjust", "approximately", "optional",
    "fresh", "dried", "ground", "whole", "sliced", "chopped", "diced", "minced",
    "finely", "roughly", "thinly", "thickly",
    "about", "allow", "already", "amount", "another", "any", "around", "away",
    "back", "become", "begin", "bit", "bottom", "carefully", "center", "change",
    "check", "completely", "continue", "create", "dark", "desired", "different",
    "done", "down", "drop", "dry", "each", "easily", "either", "else", "empty",
    "end", "enough", "entire", "especially", "even", "evenly", "every", "exactly",
    "example", "extra", "feel", "few", "fill", "final", "finally", "fine", "firm",
    "first", "follow", "following", "food", "form", "forward", "full", "fully",
    "gentle", "gently", "give", "golden", "good", "gradually", "great", "half",
    "hand", "handle", "hard", "have", "head", "heavy", "help", "high", "hold",
    "hot", "if", "immediately", "important", "increase", "inside", "instead",
    "keep", "kind", "knife", "know", "last", "layer", "least", "leave", "left",
    "less", "let", "level", "light", "lightly", "liquid", "little", "long", "longer",
    "look", "loose", "low", "main", "many", "match", "matter", "maximum", "mean",
    "means", "measure", "melted", "metal", "middle", "might", "minimum",
    "moderate", "moist", "moment", "more", "most", "move", "name", "near",
    "necessary", "next", "nice", "non", "normal", "note", "now", "number",
    "often", "once", "one", "only", "onto", "open", "order", "original", "other",
    "out", "outside", "over", "own", "part", "particular", "particularly", "paste",
    "perfect", "perfectly", "piece", "pieces", "plastic", "point", "possible",
    "prefer", "prepare", "prepared", "press", "prevent", "process", "properly",
    "provide", "pull", "push", "put", "quality", "quick", "quickly", "quite",
    "rack", "raise", "range", "rapidly", "rather", "ready", "remaining",
    "repeat", "rest", "result", "return", "right", "rise", "roll", "round",
    "run", "same", "sauce", "save", "say", "scoop", "scrape", "second", "see",
    "seem", "send", "separate", "several", "shape", "sharp", "short", "should",
    "show", "side", "sides", "similar", "simple", "simply", "since", "single",
    "size", "slightly", "slow", "slowly", "smooth", "so", "soft", "soften",
    "solid", "some", "soon", "source", "space", "special", "spoon", "spread",
    "stand", "start", "stay", "steady", "stick", "still", "stop", "store",
    "straight", "style", "such", "sufficient", "surface", "take", "tall",
    "tend", "than", "that", "the", "their", "them", "then", "there", "these",
    "thick", "thin", "thing", "think", "this", "those", "though", "three",
    "through", "throughout", "tie", "tight", "tightly", "till", "time", "tip",
    "to", "together", "too", "top", "total", "touch", "toward", "transfer",
    "trap", "trim", "true", "try", "turn", "twice", "two", "type", "under",
    "uniform", "unique", "unless", "up", "upon", "using", "usual", "usually",
    "value", "various", "very", "view", "volume", "wait", "warm", "watch",
    "way", "wet", "what", "when", "where", "which", "while", "white", "wide",
    "will", "wipe", "with", "within", "without", "wood", "wooden", "work",
    "wrap", "yellow", "yet", "yield", "you", "your"
}

# --- Initialization ---
# Ensure the results directory exists when this config is imported
if not os.path.exists(RESULTS_DIR):
    try:
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    except OSError as e:
        print(f"Error creating results directory {RESULTS_DIR}: {e}")
