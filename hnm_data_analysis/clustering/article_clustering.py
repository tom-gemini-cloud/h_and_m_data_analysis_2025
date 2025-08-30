"""
Article Clustering Module

This module provides comprehensive clustering capabilities for H&M articles using
pre-computed feature matrices (e.g., text TF-IDF/SVD embeddings). It supports multiple
clustering algorithms and evaluation metrics optimised for retail product analysis.

Key Features:
- K-means clustering with automatic k selection
- Hierarchical clustering with dendrograms
- DBSCAN for density-based clustering
- Gaussian Mixture Models for probabilistic clustering
- Cluster evaluation and interpretation tools
- Flexible feature loading from .npy (dense), .npz (sparse TF-IDF), or .parquet/.csv (with
  feature columns and an `article_id` column, e.g., SVD embeddings saved by the text vectoriser)
"""

from __future__ import annotations

import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""
    algorithm: str  # 'kmeans', 'hierarchical', 'dbscan', 'gmm'
    n_clusters: Optional[int] = None  # For algorithms that need it
    random_state: int = 42
    
    # K-means specific
    kmeans_init: str = "k-means++"
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10
    
    # DBSCAN specific
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    dbscan_metric: str = "euclidean"  # e.g., 'euclidean', 'cosine', 'hamming'
    dbscan_p: Optional[int] = None     # Minkowski p (used when metric='minkowski')
    
    # Hierarchical specific
    linkage: str = "ward"
    
    # GMM specific
    gmm_covariance_type: str = "full"
    gmm_max_iter: int = 100


@dataclass
class ClusteringResults:
    """Container for clustering results and metadata."""
    labels: np.ndarray
    algorithm: str
    n_clusters: int
    article_ids: List[Any]
    feature_shape: Tuple[int, int]
    
    # Evaluation metrics
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None
    
    # Algorithm-specific results
    cluster_centres: Optional[np.ndarray] = None  # K-means
    probabilities: Optional[np.ndarray] = None   # GMM
    
    # Interpretability
    cluster_summaries: Optional[Dict[int, Dict[str, Any]]] = None


class ArticleClusterer:
    """
    Comprehensive clustering for H&M articles using combined features.
    
    This class provides multiple clustering algorithms and evaluation tools
    for analysing article similarities using both text descriptions and
    categorical product attributes.
    """
    
    def __init__(
        self,
        features_path: Optional[str] = None,
        article_ids_path: Optional[str] = None,
        articles_metadata_path: Optional[str] = None,
    ):
        """
        Initialise the article clusterer.
        
        Args:
            features_path: Path to combined feature matrix (.npy file)
            article_ids_path: Path to article ID index (.csv file)
            articles_metadata_path: Path to original articles data for interpretation
        """
        self.features_path = features_path
        self.article_ids_path = article_ids_path
        self.articles_metadata_path = articles_metadata_path
        
        self.features: Optional[np.ndarray] = None
        self.article_ids: Optional[List[Any]] = None
        self.articles_metadata: Optional[pl.DataFrame] = None
        
        self.clustering_model: Optional[Any] = None
        self.results: Optional[ClusteringResults] = None
        
    def load_features(self, features_path: Optional[str] = None,
                      article_ids_path: Optional[str] = None) -> Tuple[np.ndarray, List[Any]]:
        """Load feature matrix and corresponding article IDs.

        Supports the following formats:
        - .npy: Dense NumPy array (requires `article_ids_path` CSV mapping)
        - .npz: Sparse matrix saved by scipy.sparse (will be densified; prefer SVD for efficiency)
        - .parquet/.csv: Tabular file containing an `article_id` column and feature columns
          (e.g., SVD embeddings `svd_###`). In this case, `article_ids_path` is optional.
        """
        features_path = features_path or self.features_path
        article_ids_path = article_ids_path or self.article_ids_path

        if features_path is None:
            raise ValueError("features_path must be provided")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}\n"
                                   "If you intended to use SVD embeddings, ensure the file exists "
                                   "(e.g., data/processed/features/svd_embeddings.parquet).")

        _, ext = os.path.splitext(features_path)
        ext = ext.lower()

        # Case 1: Dense .npy matrix
        if ext == ".npy":
            self.features = np.load(features_path)
            if article_ids_path is None:
                raise ValueError("article_ids_path must be provided when loading from .npy")
            if not os.path.exists(article_ids_path):
                raise FileNotFoundError(f"Article IDs file not found: {article_ids_path}")
            ids_df = pl.read_csv(article_ids_path)
            self.article_ids = ids_df.get_column("article_id").to_list()

        # Case 2: Sparse .npz matrix (densify with warning)
        elif ext == ".npz":
            try:
                from scipy.sparse import load_npz
            except Exception as exc:
                raise ImportError("scipy is required to load .npz sparse matrices") from exc

            sparse_matrix = load_npz(features_path)
            warnings.warn(
                "Loading a sparse .npz matrix; converting to dense may be memory intensive. "
                "Prefer SVD embeddings for clustering."
            )
            self.features = sparse_matrix.toarray()
            if article_ids_path is None:
                raise ValueError("article_ids_path must be provided when loading from .npz")
            if not os.path.exists(article_ids_path):
                raise FileNotFoundError(f"Article IDs file not found: {article_ids_path}")
            ids_df = pl.read_csv(article_ids_path)
            self.article_ids = ids_df.get_column("article_id").to_list()

        # Case 3: Tabular embeddings (.parquet or .csv) with article_id
        elif ext in {".parquet", ".csv"}:
            if ext == ".parquet":
                df = pl.read_parquet(features_path)
            else:
                df = pl.read_csv(features_path)

            if "article_id" not in df.columns:
                raise ValueError("Expected an 'article_id' column in the features file")

            # Choose feature columns: prefer columns starting with 'svd_' if present; otherwise
            # use all numeric columns except the id column.
            svd_cols = [c for c in df.columns if c.startswith("svd_")]
            if svd_cols:
                feature_cols = sorted(svd_cols)
            else:
                feature_cols = [
                    c for c, dt in zip(df.columns, df.dtypes)
                    if c != "article_id" and dt in {pl.Float64, pl.Float32, pl.Int64, pl.Int32}
                ]
                if not feature_cols:
                    raise ValueError("No numeric feature columns found in the features file")

            self.article_ids = df.get_column("article_id").to_list()
            # Convert to float32 to reduce memory footprint
            self.features = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)

        else:
            raise ValueError(f"Unsupported features file extension: {ext}")

        print(f"Loaded features: {self.features.shape[0]:,} articles x {self.features.shape[1]:,} features")

        if self.article_ids is None:
            raise ValueError("Failed to load article IDs")
        if len(self.article_ids) != self.features.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.article_ids)} article IDs vs {self.features.shape[0]} feature rows"
            )

        return self.features, self.article_ids
        
    def load_articles_metadata(self, metadata_path: Optional[str] = None) -> pl.DataFrame:
        """Load original articles data for cluster interpretation."""
        metadata_path = metadata_path or self.articles_metadata_path
        
        if metadata_path is None:
            warnings.warn("No articles metadata path provided. Cluster interpretation will be limited.")
            return None
            
        if not os.path.exists(metadata_path):
            warnings.warn(f"Articles metadata file not found: {metadata_path}")
            return None
            
        if metadata_path.endswith(".parquet"):
            self.articles_metadata = pl.read_parquet(metadata_path)
        else:
            self.articles_metadata = pl.read_csv(metadata_path)
            
        print(f"Loaded articles metadata: {self.articles_metadata.height:,} articles")
        return self.articles_metadata
        
    def find_optimal_k(self, k_range: Tuple[int, int] = (2, 20), 
                      algorithm: str = "kmeans", 
                      method: str = "elbow") -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters using elbow method or silhouette analysis.
        
        Args:
            k_range: Range of k values to test (min, max)
            algorithm: Clustering algorithm to use for evaluation
            method: Method for k selection ('elbow', 'silhouette')
            
        Returns:
            Tuple of (optimal_k, scores_dict)
        """
        if self.features is None:
            raise ValueError("Features not loaded. Call load_features() first.")
            
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        scores = {}
        
        print(f"Finding optimal k using {method} method with {algorithm}...")
        
        for k in k_values:
            if algorithm == "kmeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(self.features)
                
                if method == "elbow":
                    # Use inertia for elbow method
                    scores[k] = model.inertia_
                elif method == "silhouette":
                    scores[k] = silhouette_score(self.features, labels)
            elif algorithm == "gmm":
                model = GaussianMixture(n_components=k, random_state=42)
                labels = model.fit_predict(self.features)
                
                if method == "elbow":
                    # Use AIC for elbow method
                    scores[k] = model.aic(self.features)
                elif method == "silhouette":
                    scores[k] = silhouette_score(self.features, labels)
            else:
                raise ValueError(f"Optimal k finding not supported for algorithm: {algorithm}")
                
            print(f"k={k}: {method} score = {scores[k]:.4f}")
            
        # Find optimal k
        if method == "elbow":
            # For elbow method, look for the "elbow" point (inflection)
            # For simplicity, use the point with maximum second derivative
            k_values_list = list(k_values)
            scores_list = [scores[k] for k in k_values_list]
            
            if len(scores_list) >= 3:
                # Calculate second derivatives
                second_derivatives = []
                for i in range(1, len(scores_list) - 1):
                    second_deriv = scores_list[i-1] - 2*scores_list[i] + scores_list[i+1]
                    second_derivatives.append(second_deriv)
                
                # Find the point with maximum second derivative (elbow)
                optimal_idx = np.argmax(second_derivatives) + 1  # +1 because we started from index 1
                optimal_k = k_values_list[optimal_idx]
            else:
                optimal_k = k_min
        elif method == "silhouette":
            # For silhouette, higher is better
            optimal_k = max(scores.keys(), key=lambda k: scores[k])
            
        print(f"Optimal k selected: {optimal_k}")
        return optimal_k, scores
        
    def plot_k_selection(self, scores: Dict[int, float], method: str, 
                        optimal_k: int, save_path: Optional[str] = None) -> None:
        """Plot k selection curve."""
        k_values = sorted(scores.keys())
        score_values = [scores[k] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, score_values, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel(f'{method.title()} Score')
        plt.title(f'Optimal Number of Clusters - {method.title()} Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            # Ensure parent directory exists before saving
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved k selection plot to: {save_path}")
        plt.show()
        
    def cluster(self, config: ClusteringConfig, 
                auto_k: bool = False) -> ClusteringResults:
        """
        Perform clustering with specified configuration.
        
        Args:
            config: Clustering configuration
            auto_k: Whether to automatically find optimal k (for applicable algorithms)
            
        Returns:
            ClusteringResults object with labels and metadata
        """
        if self.features is None:
            raise ValueError("Features not loaded. Call load_features() first.")
            
        # Auto-select k if requested
        if auto_k and config.algorithm in ["kmeans", "gmm"]:
            optimal_k, _ = self.find_optimal_k(algorithm=config.algorithm)
            config.n_clusters = optimal_k
            print(f"Auto-selected k = {optimal_k} for {config.algorithm}")
            
        # Initialise clustering model
        if config.algorithm == "kmeans":
            if config.n_clusters is None:
                raise ValueError("n_clusters must be specified for K-means")
            model = KMeans(
                n_clusters=config.n_clusters,
                init=config.kmeans_init,
                max_iter=config.kmeans_max_iter,
                n_init=config.kmeans_n_init,
                random_state=config.random_state
            )
        elif config.algorithm == "hierarchical":
            if config.n_clusters is None:
                raise ValueError("n_clusters must be specified for hierarchical clustering")
            model = AgglomerativeClustering(
                n_clusters=config.n_clusters,
                linkage=config.linkage
            )
        elif config.algorithm == "dbscan":
            model = DBSCAN(
                eps=config.dbscan_eps,
                min_samples=config.dbscan_min_samples,
                metric=config.dbscan_metric,
                p=config.dbscan_p
            )
        elif config.algorithm == "gmm":
            if config.n_clusters is None:
                raise ValueError("n_components must be specified for GMM")
            model = GaussianMixture(
                n_components=config.n_clusters,
                covariance_type=config.gmm_covariance_type,
                max_iter=config.gmm_max_iter,
                random_state=config.random_state
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {config.algorithm}")
            
        # Fit model and get labels
        print(f"Performing {config.algorithm} clustering...")
        labels = model.fit_predict(self.features)
        
        # Handle DBSCAN noise points
        if config.algorithm == "dbscan":
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        else:
            n_clusters = config.n_clusters
            
        # Calculate evaluation metrics (exclude noise points for DBSCAN)
        eval_labels = labels
        eval_features = self.features
        if config.algorithm == "dbscan" and -1 in labels:
            # Exclude noise points from evaluation
            non_noise_mask = labels != -1
            eval_labels = labels[non_noise_mask]
            eval_features = self.features[non_noise_mask]
            
        silhouette = None
        calinski_harabasz = None
        davies_bouldin = None
        
        if len(set(eval_labels)) > 1:  # Need at least 2 clusters for metrics
            try:
                silhouette = silhouette_score(eval_features, eval_labels)
                calinski_harabasz = calinski_harabasz_score(eval_features, eval_labels)
                davies_bouldin = davies_bouldin_score(eval_features, eval_labels)
            except Exception as e:
                warnings.warn(f"Error calculating evaluation metrics: {e}")
                
        # Extract algorithm-specific results
        cluster_centres = None
        probabilities = None
        
        if config.algorithm == "kmeans":
            cluster_centres = model.cluster_centers_
        elif config.algorithm == "gmm":
            probabilities = model.predict_proba(self.features)
            
        # Create results object
        self.results = ClusteringResults(
            labels=labels,
            algorithm=config.algorithm,
            n_clusters=n_clusters,
            article_ids=self.article_ids,
            feature_shape=self.features.shape,
            silhouette=silhouette,
            calinski_harabasz=calinski_harabasz,
            davies_bouldin=davies_bouldin,
            cluster_centres=cluster_centres,
            probabilities=probabilities
        )
        
        self.clustering_model = model
        
        print(f"Clustering completed: {n_clusters} clusters")
        if silhouette is not None:
            print(f"Silhouette Score: {silhouette:.4f}")
        if calinski_harabasz is not None:
            print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        if davies_bouldin is not None:
            print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            
        return self.results

    def compute_k_distances(
        self,
        k: int,
        metric: str = "euclidean",
        p: Optional[int] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> np.ndarray:
        """Compute sorted k-distance profile used to choose eps for DBSCAN.

        Args:
            k: Neighbour rank to use (typically equals DBSCAN min_samples)
            metric: Distance metric (e.g., 'euclidean', 'cosine', 'hamming', 'minkowski')
            p: Minkowski power parameter if metric='minkowski'
            sample_size: If provided, randomly sample this many points for speed
            random_state: RNG seed for sampling

        Returns:
            Sorted array of k-distances (ascending)
        """
        if self.features is None:
            raise ValueError("Features not loaded. Call load_features() first.")

        rng = np.random.default_rng(random_state)
        X = self.features
        if sample_size is not None and sample_size < X.shape[0]:
            indices = rng.choice(X.shape[0], size=sample_size, replace=False)
            X = X[indices]

        # n_neighbors=k because the first neighbour is the point itself
        nn = NearestNeighbors(n_neighbors=k, metric=metric, p=p)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        # Take the k-th distance (last column)
        k_distances = distances[:, -1]
        return np.sort(k_distances)

    def plot_k_distance(
        self,
        k: int,
        metric: str = "euclidean",
        p: Optional[int] = None,
        sample_size: Optional[int] = None,
        save_path: Optional[str] = None,
        title_suffix: str = "",
    ) -> np.ndarray:
        """Plot and optionally save the sorted k-distance curve.

        Returns the sorted k-distance array for further analysis.
        """
        k_distances_sorted = self.compute_k_distances(k=k, metric=metric, p=p, sample_size=sample_size)

        plt.figure(figsize=(10, 6))
        plt.plot(k_distances_sorted)
        plt.xlabel("Points sorted by k-distance")
        plt.ylabel(f"{metric} distance to {k}-th neighbour")
        title = f"k-distance plot (k={k}, metric={metric})"
        if title_suffix:
            title += f" {title_suffix}"
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved k-distance plot to: {save_path}")
        plt.show()
        return k_distances_sorted
        
    def interpret_clusters(self, top_features: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        Interpret clusters by analysing their characteristics.
        
        Args:
            top_features: Number of top features to show per cluster
            
        Returns:
            Dictionary with cluster interpretations
        """
        if self.results is None:
            raise ValueError("No clustering results available. Call cluster() first.")
            
        cluster_summaries = {}
        
        # Get unique clusters (excluding noise for DBSCAN)
        unique_clusters = [c for c in set(self.results.labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = self.results.labels == cluster_id
            cluster_articles = [aid for i, aid in enumerate(self.article_ids) 
                              if cluster_mask[i]]
            
            summary = {
                "size": int(np.sum(cluster_mask)),
                "article_ids": cluster_articles[:20],  # Show first 20 article IDs
                "percentage": float(np.sum(cluster_mask) / len(self.results.labels) * 100)
            }
            
            # Add metadata analysis if available
            if self.articles_metadata is not None:
                # Filter metadata for this cluster
                cluster_metadata = self.articles_metadata.filter(
                    pl.col("article_id").is_in(cluster_articles)
                )
                
                if cluster_metadata.height > 0:
                    # Analyse categorical distributions
                    categorical_cols = [
                        "product_group_name", "product_type_name", 
                        "colour_group_name", "department_name",
                        "garment_group_name"
                    ]
                    
                    for col in categorical_cols:
                        if col in cluster_metadata.columns:
                            # Build a plain Python {category_value: count} dict
                            vc_df = (
                                cluster_metadata
                                .group_by(col)
                                .count()
                                .sort("count", descending=True)
                                .head(5)
                            )
                            top_values: Dict[str, int] = {
                                (str(val) if val is not None else "None"): int(cnt)
                                for val, cnt in vc_df.iter_rows()
                            }
                            summary[f"top_{col}"] = top_values
                            
            cluster_summaries[cluster_id] = summary
            
        self.results.cluster_summaries = cluster_summaries
        return cluster_summaries
        
    def visualise_clusters(self, method: str = "pca", 
                          save_path: Optional[str] = None) -> None:
        """
        Visualise clusters in 2D using dimensionality reduction.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'tsne')
            save_path: Path to save the plot
        """
        if self.results is None:
            raise ValueError("No clustering results available. Call cluster() first.")
            
        print(f"Creating 2D visualization using {method.upper()}...")
        
        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(self.features)
            title_suffix = f"(PCA: {reducer.explained_variance_ratio_.sum():.1%} variance explained)"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords_2d = reducer.fit_transform(self.features)
            title_suffix = "(t-SNE)"
        else:
            raise ValueError(f"Unsupported visualization method: {method}")
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Get unique clusters and assign colours
        unique_clusters = sorted(set(self.results.labels))
        colours = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = self.results.labels == cluster_id
            
            if cluster_id == -1:  # Noise points for DBSCAN
                plt.scatter(coords_2d[cluster_mask, 0], coords_2d[cluster_mask, 1],
                           c='black', s=20, alpha=0.5, label='Noise')
            else:
                plt.scatter(coords_2d[cluster_mask, 0], coords_2d[cluster_mask, 1],
                           c=[colours[i]], s=30, alpha=0.7, 
                           label=f'Cluster {cluster_id}')
                
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Article Clusters Visualisation {title_suffix}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            # Ensure parent directory exists before saving
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster visualization to: {save_path}")
        plt.show()
        
    def save_results(self, output_dir: str) -> None:
        """Save clustering results and interpretations."""
        if self.results is None:
            raise ValueError("No clustering results to save. Call cluster() first.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cluster labels
        labels_df = pl.DataFrame({
            "article_id": self.article_ids,
            "cluster_label": self.results.labels
        })
        labels_path = os.path.join(output_dir, "cluster_labels.csv")
        labels_df.write_csv(labels_path)
        print(f"Saved cluster labels to: {labels_path}")
        
        # Save clustering model
        model_path = os.path.join(output_dir, "clustering_model.joblib")
        joblib.dump(self.clustering_model, model_path)
        print(f"Saved clustering model to: {model_path}")
        
        # Save results metadata (cast numpy types to native Python types)
        results_dict = {
            "algorithm": str(self.results.algorithm),
            "n_clusters": int(self.results.n_clusters),
            "feature_shape": [int(self.results.feature_shape[0]), int(self.results.feature_shape[1])],
            "silhouette_score": (float(self.results.silhouette)
                                   if self.results.silhouette is not None else None),
            "calinski_harabasz_score": (float(self.results.calinski_harabasz)
                                         if self.results.calinski_harabasz is not None else None),
            "davies_bouldin_score": (float(self.results.davies_bouldin)
                                       if self.results.davies_bouldin is not None else None),
        }
        
        # Add cluster size distribution
        unique_labels, counts = np.unique(self.results.labels, return_counts=True)
        cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        results_dict["cluster_sizes"] = cluster_sizes
        
        metadata_path = os.path.join(output_dir, "clustering_results.json")
        with open(metadata_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"Saved results metadata to: {metadata_path}")
        
        # Save cluster interpretations if available
        if self.results.cluster_summaries:
            summaries_path = os.path.join(output_dir, "cluster_interpretations.json")
            with open(summaries_path, "w") as f:
                # Convert numpy types to native Python types for JSON serialisation
                serializable_summaries = {}
                for cluster_id, summary in self.results.cluster_summaries.items():
                    serializable_summary = {}
                    for key, value in summary.items():
                        if isinstance(value, (np.integer, np.floating)):
                            serializable_summary[key] = value.item()
                        elif isinstance(value, np.ndarray):
                            serializable_summary[key] = value.tolist()
                        else:
                            serializable_summary[key] = value
                    serializable_summaries[str(cluster_id)] = serializable_summary
                    
                json.dump(serializable_summaries, f, indent=2)
            print(f"Saved cluster interpretations to: {summaries_path}")


def main():
    """Example usage of article clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster articles using combined features")
    parser.add_argument("--features-path", type=str, required=True,
                       help="Path to combined features (.npy)")
    parser.add_argument("--article-ids-path", type=str, required=True,
                       help="Path to article IDs (.csv)")
    parser.add_argument("--articles-metadata-path", type=str,
                       help="Path to original articles data for interpretation")
    parser.add_argument("--output-dir", type=str, default="../results/clustering",
                       help="Output directory for clustering results")
    parser.add_argument("--algorithm", type=str, default="kmeans",
                       choices=["kmeans", "hierarchical", "dbscan", "gmm"],
                       help="Clustering algorithm")
    parser.add_argument("--n-clusters", type=int, default=None,
                       help="Number of clusters (auto-detected if not provided)")
    parser.add_argument("--auto-k", action="store_true",
                       help="Automatically find optimal number of clusters")
    parser.add_argument("--visualize", action="store_true",
                       help="Create cluster visualisations")
    
    args = parser.parse_args()
    
    # Initialise clusterer
    clusterer = ArticleClusterer(
        features_path=args.features_path,
        article_ids_path=args.article_ids_path,
        articles_metadata_path=args.articles_metadata_path
    )
    
    # Load data
    clusterer.load_features()
    if args.articles_metadata_path:
        clusterer.load_articles_metadata()
        
    # Create clustering configuration
    config = ClusteringConfig(
        algorithm=args.algorithm,
        n_clusters=args.n_clusters
    )
    
    # Perform clustering
    results = clusterer.cluster(config, auto_k=args.auto_k or args.n_clusters is None)
    
    # Interpret clusters
    clusterer.interpret_clusters()
    
    # Create visualisations if requested
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        clusterer.visualise_clusters(
            method="pca",
            save_path=os.path.join(args.output_dir, "clusters_pca.png")
        )
        clusterer.visualise_clusters(
            method="tsne", 
            save_path=os.path.join(args.output_dir, "clusters_tsne.png")
        )
        
    # Save results
    clusterer.save_results(args.output_dir)
    
    print(f"\nClustering completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
