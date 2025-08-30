"""
Example: Using BERT embeddings for article feature engineering

This example shows the equivalent BERT-based approach to the TF-IDF feature engineering.
"""

import sys
sys.path.append('../') 

# TF-IDF approach (existing)
from hnm_data_analysis.feature_engineering.articles_text_tfidf import ArticleDescriptionVectoriser

print("=== TF-IDF Approach ===")
vec = ArticleDescriptionVectoriser(
    input_path="../data/cleaned/articles_last_3_months_cleaned.parquet",
    language="en",
    use_lemmatise=True,
    use_stem=False,
)
tfidf, svd = vec.process(
    output_dir="../data/processed/features",
    include_svd=True,
    svd_components=200,
    max_features=30000, min_df=5, max_df=0.8, ngram_range=(1,2),
)

print("\n" + "="*50 + "\n")

# BERT approach (new equivalent)
from hnm_data_analysis.feature_engineering.articles_text_bert import ArticleDescriptionBertEmbedder

print("=== BERT Approach ===")
bert_embedder = ArticleDescriptionBertEmbedder(
    input_path="../data/cleaned/articles_last_3_months_cleaned.parquet",
    model_name="all-MiniLM-L6-v2",  # Fast, good quality model
    batch_size=32,
    device="auto",  # Uses GPU if available
)
bert_embeddings, pca_embeddings = bert_embedder.process(
    output_dir="../data/processed/features/bert",
    include_pca=True,
    pca_components=200,
    pca_normalize=True,
    show_progress_bar=True,
)

print(f"\nBERT embeddings shape: {bert_embeddings.shape}")
print(f"PCA embeddings shape: {pca_embeddings.shape if pca_embeddings is not None else 'None'}")

# Alternative BERT models you can try:
print("\n=== Alternative BERT Models ===")
models_to_try = [
    "all-MiniLM-L6-v2",      # Fast, 384 dimensions
    "all-mpnet-base-v2",     # Higher quality, 768 dimensions  
    "all-MiniLM-L12-v2",     # Better quality, 384 dimensions
    "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
]

for model_name in models_to_try:
    print(f"- {model_name}")

print("\n=== Usage Notes ===")
print("1. BERT embeddings capture semantic meaning, not just word frequency")
print("2. They should give better clustering results than TF-IDF")
print("3. The 'all-MiniLM-L6-v2' model is a good balance of speed and quality")
print("4. Use 'all-mpnet-base-v2' for higher quality but slower processing")
print("5. The PCA embeddings can be used directly for clustering")
