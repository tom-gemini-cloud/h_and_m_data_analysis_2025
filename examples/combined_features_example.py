"""
Example: Using CombinedFeaturesEngine

This example demonstrates how to use the CombinedFeaturesEngine to combine
SVD text embeddings with categorical article features.
"""

from hnm_data_analysis.feature_engineering import CombinedFeaturesEngine


def example_basic_usage():
    """Basic usage with default settings."""
    print("=== Basic Usage Example ===")
    
    # Create engine with default paths
    engine = CombinedFeaturesEngine()
    
    # Generate combined features
    output_path = engine.create_combined_features()
    
    print(f"Combined features saved to: {output_path}")


def example_custom_paths():
    """Example with custom input/output paths."""
    print("\n=== Custom Paths Example ===")
    
    # Create engine with custom paths
    engine = CombinedFeaturesEngine(
        svd_embeddings_path="data/processed/features/svd_embeddings.parquet",
        cleaned_articles_path="data/cleaned/articles_last_3_months_cleaned.parquet",
        output_dir="data/features/custom_output"
    )
    
    # Generate features (comment out to avoid running)
    # output_path = engine.create_combined_features()
    # print(f"Custom combined features saved to: {output_path}")
    
    print("Custom engine configured (commented out execution)")


def example_step_by_step():
    """Example showing step-by-step processing."""
    print("\n=== Step-by-Step Example ===")
    
    engine = CombinedFeaturesEngine()
    
    # Step 1: Load datasets
    svd_df, articles_df = engine.load_datasets()
    print(f"Loaded {svd_df.shape[0]} SVD embeddings and {articles_df.shape[0]} articles")
    
    # Step 2: Select categorical features
    categorical_features = engine.select_categorical_features()
    print(f"Selected {len(categorical_features)} categorical features")
    
    # Step 3: Encode categorical features (comment out for demo)
    # encoded_categorical, feature_names = engine.encode_categorical_features()
    # print(f"Encoded to {encoded_categorical.shape[1]} dimensions")
    
    print("Demonstrated step-by-step loading (commented out full processing)")


if __name__ == "__main__":
    # Run only the basic example to avoid creating duplicate files
    example_basic_usage()
    example_custom_paths()
    example_step_by_step()