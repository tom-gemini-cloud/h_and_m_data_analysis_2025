"""
Combined Features Engineering Module

This module combines SVD text embeddings with categorical features from cleaned articles
metadata to create comprehensive features for article clustering and analysis.

The module processes:
- SVD embeddings (200 dimensions) from text descriptions
- Categorical product attributes (encoded appropriately)
- Produces scaled, combined feature matrix

Key Features:
- Mixed encoding strategy (one-hot for low cardinality, label for high cardinality)
- Feature scaling and normalisation
- Memory-efficient processing with Polars
- Comprehensive validation and error handling

Usage:
    from hnm_data_analysis.feature_engineering.combined_features import CombinedFeaturesEngine
    
    engine = CombinedFeaturesEngine()
    combined_features_path = engine.create_combined_features()
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class CombinedFeaturesEngine:
    """
    Engine for combining SVD embeddings with categorical article features.
    
    This class handles the complete pipeline for creating combined features:
    1. Load SVD embeddings and cleaned articles metadata
    2. Select and encode categorical features appropriately
    3. Combine SVD and categorical features
    4. Scale features for machine learning
    5. Save processed features in multiple formats
    """
    
    def __init__(
        self,
        svd_embeddings_path: Optional[str] = None,
        cleaned_articles_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialise the combined features engine.
        
        Args:
            svd_embeddings_path: Path to SVD embeddings file (.parquet)
            cleaned_articles_path: Path to cleaned articles metadata (.parquet) 
            output_dir: Directory to save combined features (defaults to data/features)
        """
        # Default paths based on project structure
        self.svd_embeddings_path = svd_embeddings_path or "data/processed/features/svd_embeddings.parquet"
        self.cleaned_articles_path = cleaned_articles_path or "data/cleaned/articles_last_3_months_cleaned.parquet"
        self.output_dir = output_dir or "data/features"
        
        # Internal state
        self.svd_df: Optional[pl.DataFrame] = None
        self.articles_df: Optional[pl.DataFrame] = None
        self.categorical_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        
    def load_datasets(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load SVD embeddings and cleaned articles datasets.
        
        Returns:
            Tuple of (svd_dataframe, articles_dataframe)
            
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If datasets have incompatible structures
        """
        print("Loading datasets...")
        
        # Load SVD embeddings
        if not os.path.exists(self.svd_embeddings_path):
            raise FileNotFoundError(f"SVD embeddings not found: {self.svd_embeddings_path}")
            
        self.svd_df = pl.read_parquet(self.svd_embeddings_path)
        print(f"Loaded SVD embeddings: {self.svd_df.shape}")
        
        # Validate SVD structure
        if "article_id" not in self.svd_df.columns:
            raise ValueError("SVD embeddings must contain 'article_id' column")
            
        svd_feature_cols = [col for col in self.svd_df.columns if col.startswith("svd_")]
        if len(svd_feature_cols) == 0:
            raise ValueError("No SVD feature columns found (expected columns starting with 'svd_')")
        print(f"Found {len(svd_feature_cols)} SVD features")
        
        # Load cleaned articles
        if not os.path.exists(self.cleaned_articles_path):
            raise FileNotFoundError(f"Cleaned articles not found: {self.cleaned_articles_path}")
            
        self.articles_df = pl.read_parquet(self.cleaned_articles_path)
        print(f"Loaded cleaned articles: {self.articles_df.shape}")
        
        # Validate articles structure
        if "article_id" not in self.articles_df.columns:
            raise ValueError("Articles metadata must contain 'article_id' column")
            
        return self.svd_df, self.articles_df
        
    def select_categorical_features(self) -> List[str]:
        """
        Select relevant categorical features for clustering.
        
        Chooses features that are meaningful for product similarity and clustering,
        excluding technical identifiers and low-information columns.
        
        Returns:
            List of selected categorical feature column names
        """
        # Core categorical features for product clustering
        candidate_features = [
            'product_type_name',        # Type of product (dress, shirt, etc.)
            'product_group_name',       # Product group classification
            'colour_group_name',        # Colour category
            'department_name',          # Department (Ladies, Men, Kids, etc.)
            'garment_group_name',       # Garment category
            'perceived_colour_value_name',  # Colour perception (Light, Dark, etc.)
            'perceived_colour_master_name', # Master colour category
            'graphical_appearance_name',    # Pattern/appearance
            'index_name',               # Fashion index category
            'section_name',             # Store section
        ]
        
        # Filter to features that exist in the dataset
        available_features = [f for f in candidate_features if f in self.articles_df.columns]
        
        # Additional validation - check for sufficient diversity
        final_features = []
        for feature in available_features:
            unique_count = self.articles_df[feature].n_unique()
            if unique_count > 1:  # Must have more than 1 unique value
                final_features.append(feature)
            else:
                warnings.warn(f"Skipping feature '{feature}' - only {unique_count} unique value(s)")
        
        self.categorical_features = final_features
        print(f"Selected categorical features: {final_features}")
        
        return final_features
        
    def encode_categorical_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Encode categorical features using mixed strategy.
        
        Uses one-hot encoding for low cardinality features and label encoding
        for high cardinality features to balance interpretability and efficiency.
        
        Returns:
            Tuple of (encoded_features_array, feature_names_list)
        """
        print("Encoding categorical features...")
        
        if not self.categorical_features:
            raise ValueError("No categorical features selected. Call select_categorical_features() first.")
            
        # Prepare subset with only categorical features
        articles_subset = self.articles_df.select(['article_id'] + self.categorical_features)
        
        # Convert to pandas for sklearn preprocessing
        articles_pd = articles_subset.to_pandas()
        
        # Categorise features by cardinality
        low_cardinality_features = []
        high_cardinality_features = []
        cardinality_threshold = 15
        
        for col in self.categorical_features:
            unique_count = articles_pd[col].nunique()
            if unique_count <= cardinality_threshold:
                low_cardinality_features.append(col)
            else:
                high_cardinality_features.append(col)
        
        print(f"Low cardinality (one-hot, <={cardinality_threshold}): {low_cardinality_features}")
        print(f"High cardinality (label, >{cardinality_threshold}): {high_cardinality_features}")
        
        # Encode features
        encoded_parts = []
        feature_names = []
        
        # One-hot encode low cardinality features
        if low_cardinality_features:
            onehot = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            onehot_encoded = onehot.fit_transform(articles_pd[low_cardinality_features])
            encoded_parts.append(onehot_encoded)
            
            onehot_names = onehot.get_feature_names_out(low_cardinality_features)
            feature_names.extend(onehot_names)
        
        # Label encode high cardinality features
        if high_cardinality_features:
            label_encoded = np.zeros((len(articles_pd), len(high_cardinality_features)))
            for i, col in enumerate(high_cardinality_features):
                le = LabelEncoder()
                # Handle missing values by replacing with 'Unknown'
                label_encoded[:, i] = le.fit_transform(articles_pd[col].fillna('Unknown'))
            
            encoded_parts.append(label_encoded)
            feature_names.extend([f"{col}_encoded" for col in high_cardinality_features])
        
        # Combine encoded features
        if len(encoded_parts) > 1:
            encoded_features = np.hstack(encoded_parts)
        elif len(encoded_parts) == 1:
            encoded_features = encoded_parts[0]
        else:
            raise ValueError("No features to encode")
        
        print(f"Encoded categorical features shape: {encoded_features.shape}")
        return encoded_features, feature_names
        
    def combine_with_svd_features(
        self, 
        encoded_categorical: np.ndarray, 
        categorical_feature_names: List[str]
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Combine SVD embeddings with encoded categorical features.
        
        Args:
            encoded_categorical: Encoded categorical features array
            categorical_feature_names: Names of categorical features
            
        Returns:
            Tuple of (combined_dataframe, all_feature_names)
        """
        print("Combining SVD and categorical features...")
        
        # Get article IDs for categorical features (same order as encoded array)
        articles_subset = self.articles_df.select(['article_id'] + self.categorical_features)
        categorical_article_ids = articles_subset['article_id'].to_list()
        
        # Create categorical features dataframe
        categorical_data = {'article_id': categorical_article_ids}
        for i, name in enumerate(categorical_feature_names):
            categorical_data[name] = encoded_categorical[:, i]
        
        categorical_df = pl.DataFrame(categorical_data)
        
        # Merge with SVD features (inner join to keep only articles with both)
        combined_df = self.svd_df.join(categorical_df, on='article_id', how='inner')
        
        # Get all feature column names
        svd_feature_cols = [col for col in self.svd_df.columns if col.startswith('svd_')]
        all_feature_names = svd_feature_cols + categorical_feature_names
        
        print(f"Combined features shape: {combined_df.shape}")
        print(f"Articles with both SVD and categorical features: {len(combined_df)}")
        print(f"Total features: {len(all_feature_names)} ({len(svd_feature_cols)} SVD + {len(categorical_feature_names)} categorical)")
        
        return combined_df, all_feature_names
        
    def scale_features(
        self, 
        combined_df: pl.DataFrame, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale combined features for machine learning.
        
        Args:
            combined_df: Combined features dataframe
            feature_names: Names of all features
            
        Returns:
            Tuple of (scaled_features_array, article_ids_array)
        """
        print("Scaling features...")
        
        # Extract features and article IDs
        article_ids = combined_df['article_id'].to_numpy()
        features = combined_df.select(feature_names).to_numpy()
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        
        print(f"Scaled features shape: {scaled_features.shape}")
        return scaled_features, article_ids
        
    def save_combined_features(
        self, 
        scaled_features: np.ndarray, 
        article_ids: np.ndarray, 
        feature_names: List[str]
    ) -> str:
        """
        Save combined features in multiple formats.
        
        Args:
            scaled_features: Scaled feature matrix
            article_ids: Corresponding article IDs
            feature_names: Names of all features
            
        Returns:
            Path to the main parquet file
        """
        print("Saving combined features...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create combined dataframe
        feature_data = {'article_id': article_ids}
        for i, name in enumerate(feature_names):
            feature_data[name] = scaled_features[:, i]
        
        combined_df = pl.DataFrame(feature_data)
        
        # Save as parquet (primary format)
        parquet_path = os.path.join(self.output_dir, "combined_article_features.parquet")
        combined_df.write_parquet(parquet_path)
        print(f"Saved combined features to: {parquet_path}")
        
        # Save as numpy array for direct ML use
        features_array_path = os.path.join(self.output_dir, "combined_features_array.npy")
        np.save(features_array_path, scaled_features)
        print(f"Saved features array to: {features_array_path}")
        
        # Save article IDs index
        ids_path = os.path.join(self.output_dir, "combined_article_ids.csv")
        pl.DataFrame({'article_id': article_ids}).write_csv(ids_path)
        print(f"Saved article IDs to: {ids_path}")
        
        # Save feature names for reference
        feature_names_path = os.path.join(self.output_dir, "combined_feature_names.txt")
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"Saved feature names to: {feature_names_path}")
        
        # Save preprocessing objects
        if self.scaler:
            scaler_path = os.path.join(self.output_dir, "feature_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to: {scaler_path}")
        
        return parquet_path
        
    def create_combined_features(self) -> str:
        """
        Complete pipeline to create combined features.
        
        Executes the full feature combination workflow:
        1. Load datasets
        2. Select categorical features  
        3. Encode categorical features
        4. Combine with SVD features
        5. Scale features
        6. Save results
        
        Returns:
            Path to the saved combined features parquet file
        """
        print("=== Creating Combined Article Features ===")
        
        # Load datasets
        self.load_datasets()
        
        # Select and encode categorical features
        self.select_categorical_features()
        encoded_categorical, categorical_feature_names = self.encode_categorical_features()
        
        # Combine with SVD features
        combined_df, all_feature_names = self.combine_with_svd_features(
            encoded_categorical, categorical_feature_names
        )
        
        # Scale features
        scaled_features, article_ids = self.scale_features(combined_df, all_feature_names)
        
        # Save combined features
        output_path = self.save_combined_features(scaled_features, article_ids, all_feature_names)
        
        # Summary
        n_svd_features = len([name for name in all_feature_names if name.startswith('svd_')])
        n_categorical_features = len(categorical_feature_names)
        
        print(f"\n=== Feature Combination Completed Successfully! ===")
        print(f"Final feature matrix: {scaled_features.shape}")
        print(f"Features breakdown:")
        print(f"  - SVD text embeddings: {n_svd_features} dimensions")
        print(f"  - Categorical features: {n_categorical_features} dimensions")
        print(f"  - Total: {scaled_features.shape[1]} dimensions")
        print(f"Articles processed: {len(article_ids):,}")
        print(f"Output file: {output_path}")
        
        return output_path


def main():
    """Command-line interface for combined features generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine SVD embeddings with categorical article features"
    )
    parser.add_argument(
        "--svd-embeddings-path", 
        type=str, 
        default="data/processed/features/svd_embeddings.parquet",
        help="Path to SVD embeddings file"
    )
    parser.add_argument(
        "--cleaned-articles-path", 
        type=str, 
        default="data/cleaned/articles_last_3_months_cleaned.parquet",
        help="Path to cleaned articles metadata file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/features",
        help="Output directory for combined features"
    )
    
    args = parser.parse_args()
    
    # Create combined features engine
    engine = CombinedFeaturesEngine(
        svd_embeddings_path=args.svd_embeddings_path,
        cleaned_articles_path=args.cleaned_articles_path,
        output_dir=args.output_dir
    )
    
    # Generate combined features
    output_path = engine.create_combined_features()
    
    print(f"\nCombined features saved to: {output_path}")


if __name__ == "__main__":
    main()