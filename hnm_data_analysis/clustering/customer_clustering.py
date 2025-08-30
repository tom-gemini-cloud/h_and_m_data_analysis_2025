"""
H&M Customer Clustering Module

This module provides comprehensive customer segmentation capabilities using multiple clustering approaches
optimised for H&M fashion retail data analysis.
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Advanced customer clustering for H&M retail analytics with multiple segmentation approaches.
    
    Features three clustering strategies:
    1. RFM-based clustering - Business interpretable segments
    2. Behavioural clustering - Full feature comprehensive analysis  
    3. Hybrid clustering - Balanced approach combining RFM with preferences
    """
    
    def __init__(self, data_path: str, sample_size: Optional[int] = None):
        """
        Initialise customer segmentation with feature data.
        
        Args:
            data_path: Path to customers_features_final.parquet
            sample_size: Optional sample size for memory optimisation
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.df = None
        self.processed_data = {}
        self.models = {}
        self.cluster_results = {}
        
    def load_data(self) -> pl.DataFrame:
        """Load and optionally sample customer features data."""
        print(f"Loading customer features from {self.data_path}")
        
        df = pl.read_parquet(self.data_path)
        
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, shuffle=True, seed=42)
            print(f"Sampled {self.sample_size:,} customers from {len(pl.read_parquet(self.data_path)):,} total")
        
        self.df = df
        print(f"Data loaded: {df.shape[0]:,} customers, {df.shape[1]} features")
        return df
    
    def prepare_features(self, approach: str = 'hybrid') -> pd.DataFrame:
        """
        Prepare features for clustering based on selected approach.
        
        Args:
            approach: 'rfm', 'behavioural', or 'hybrid'
        
        Returns:
            Processed feature dataframe ready for clustering
        """
        if self.df is None:
            self.load_data()
            
        # Define feature groups
        rfm_features = ['recency', 'frequency', 'monetary']
        preference_features = [
            'purchase_diversity_score', 'price_sensitivity_index', 
            'colour_preference_entropy', 'style_consistency_score'
        ]
        demographic_features = ['age', 'FN', 'Active']
        
        # Select features based on approach
        if approach == 'rfm':
            numeric_features = rfm_features
            print("Using RFM-based clustering approach")
        elif approach == 'preference':
            numeric_features = preference_features
            print("Using preference-only clustering approach")
        elif approach == 'behavioural':
            numeric_features = rfm_features + preference_features + demographic_features
            print("Using comprehensive behavioural clustering approach")
        else:  # hybrid
            numeric_features = rfm_features + preference_features
            print("Using hybrid RFM + preferences clustering approach")
            
        # Extract numeric features
        df_numeric = self.df.select(['customer_id'] + numeric_features).to_pandas()
        
        # Handle categorical features if behavioural approach
        if approach == 'behavioural':
            # Add encoded categorical features
            categorical_cols = ['club_member_status', 'fashion_news_frequency']
            df_cat = self.df.select(['customer_id'] + categorical_cols).to_pandas()
            
            # Label encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_cat[f'{col}_encoded'] = le.fit_transform(df_cat[col])
                label_encoders[col] = le
                
            # Merge numeric and encoded categorical
            df_processed = df_numeric.merge(df_cat[['customer_id'] + [f'{col}_encoded' for col in categorical_cols]], 
                                          on='customer_id')
            feature_cols = numeric_features + [f'{col}_encoded' for col in categorical_cols]
        else:
            df_processed = df_numeric.copy()
            feature_cols = numeric_features
        
        # Scale features
        scaler = StandardScaler()
        df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
        
        # Store preprocessing objects
        self.processed_data[approach] = {
            'data': df_processed,
            'feature_cols': feature_cols,
            'scaler': scaler,
            'label_encoders': label_encoders if approach == 'behavioural' else None
        }
        
        print(f"Features prepared: {len(feature_cols)} features for {len(df_processed):,} customers")
        return df_processed
    
    def find_optimal_clusters(self, approach: str = 'hybrid', max_k: int = 10, 
                             sample_for_silhouette: int = 10000) -> Dict:
        """
        Find optimal number of clusters using multiple evaluation metrics.
        
        Args:
            approach: Clustering approach to use
            max_k: Maximum number of clusters to test
            sample_for_silhouette: Sample size for silhouette calculation (for performance)
            
        Returns:
            Dictionary with optimal cluster counts and evaluation scores
        """
        if approach not in self.processed_data:
            self.prepare_features(approach)
            
        data = self.processed_data[approach]['data']
        feature_cols = self.processed_data[approach]['feature_cols']
        X = data[feature_cols].values
        
        # Create sample for silhouette score calculation (performance optimisation)
        if len(X) > sample_for_silhouette:
            sample_idx = np.random.choice(len(X), sample_for_silhouette, replace=False)
            X_sample = X[sample_idx]
            print(f"Using sample of {sample_for_silhouette:,} customers for silhouette calculation (performance optimisation)")
        else:
            X_sample = X
            sample_idx = np.arange(len(X))
        
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        print(f"Evaluating optimal clusters for {approach} approach...")
        
        for i, k in enumerate(k_range):
            print(f"  Testing k={k} ({i+1}/{len(k_range)})...")
            
            # K-means clustering with optimised parameters
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100, 
                          algorithm='lloyd')  # Use Lloyd algorithm for better performance
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
            
            # Silhouette score on sample only (for performance)
            sample_labels = cluster_labels[sample_idx]
            silhouette_scores.append(silhouette_score(X_sample, sample_labels))
            
        # Find optimal k using different criteria
        optimal_metrics = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_k_silhouette': k_range[np.argmax(silhouette_scores)],
            'optimal_k_calinski': k_range[np.argmax(calinski_scores)]
        }
        
        # Calculate elbow method (rate of change in inertia)
        inertia_changes = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        elbow_k = k_range[np.argmax(inertia_changes) + 1]
        optimal_metrics['optimal_k_elbow'] = elbow_k
        
        print(f"Optimal clusters - Silhouette: {optimal_metrics['optimal_k_silhouette']}, "
              f"Calinski-Harabasz: {optimal_metrics['optimal_k_calinski']}, "
              f"Elbow: {optimal_metrics['optimal_k_elbow']}")
              
        return optimal_metrics
    
    def perform_clustering(self, approach: str = 'hybrid', n_clusters: int = 5, 
                          method: str = 'kmeans') -> pd.DataFrame:
        """
        Perform customer clustering using specified approach and method.
        
        Args:
            approach: 'rfm', 'behavioural', or 'hybrid'
            n_clusters: Number of clusters
            method: 'kmeans' or 'hierarchical'
            
        Returns:
            DataFrame with customer clusters and original features
        """
        if approach not in self.processed_data:
            self.prepare_features(approach)
            
        data = self.processed_data[approach]['data']
        feature_cols = self.processed_data[approach]['feature_cols']
        X = data[feature_cols].values
        
        print(f"Performing {method} clustering with {n_clusters} clusters using {approach} approach...")
        
        # Choose clustering algorithm with optimised parameters
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, 
                          max_iter=300, algorithm='lloyd')
        else:  # hierarchical
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            
        # Fit model and predict clusters
        cluster_labels = model.fit_predict(X)
        
        # Add clusters to data
        result_df = data.copy()
        result_df[f'{approach}_cluster'] = cluster_labels
        
        # Calculate cluster metrics (with sampling for silhouette performance)
        sample_size = min(10000, len(X))
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
            labels_sample = cluster_labels[sample_idx]
            silhouette_avg = silhouette_score(X_sample, labels_sample)
        else:
            silhouette_avg = silhouette_score(X, cluster_labels)
            
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        
        print(f"Clustering complete - Silhouette Score: {silhouette_avg:.3f}, "
              f"Calinski-Harabasz Score: {calinski_score:.1f}")
        
        # Store results
        self.models[f'{approach}_{method}'] = model
        self.cluster_results[f'{approach}_{method}'] = {
            'data': result_df,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score,
            'cluster_column': f'{approach}_cluster'
        }
        
        return result_df
    
    def analyze_clusters(self, approach: str = 'hybrid', method: str = 'kmeans') -> pd.DataFrame:
        """
        Analyze cluster characteristics and business insights.
        
        Args:
            approach: Clustering approach used
            method: Clustering method used
            
        Returns:
            DataFrame with cluster profiles and statistics
        """
        key = f'{approach}_{method}'
        
        if key not in self.cluster_results:
            raise ValueError(f"No clustering results found for {key}. Run perform_clustering first.")
            
        result_data = self.cluster_results[key]['data']
        cluster_col = self.cluster_results[key]['cluster_column']
        feature_cols = self.processed_data[approach]['feature_cols']
        
        print(f"Analyzing clusters for {approach} approach...")
        
        # Use Polars for faster operations, then convert to pandas
        original_data_pl = self.df
        result_data_pl = pl.from_pandas(result_data[['customer_id', cluster_col]])
        
        # Fast join using Polars
        result_with_original_pl = result_data_pl.join(
            original_data_pl, on='customer_id', how='left'
        )
        
        # Convert to pandas for analysis (smaller subset operations)
        result_with_original = result_with_original_pl.to_pandas()
        
        # Calculate cluster profiles using vectorised operations
        cluster_profiles = []
        unique_clusters = sorted(result_with_original[cluster_col].unique())
        
        print(f"  Calculating profiles for {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            cluster_mask = result_with_original[cluster_col] == cluster_id
            cluster_data = result_with_original[cluster_mask]
            
            profile = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(result_with_original) * 100
            }
            
            # RFM metrics (always available)
            profile.update({
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(), 
                'avg_monetary': cluster_data['monetary'].mean()
            })
            
            # Additional metrics based on approach
            if approach in ['preference', 'hybrid', 'behavioural']:
                profile.update({
                    'avg_diversity_score': cluster_data['purchase_diversity_score'].mean(),
                    'avg_price_sensitivity': cluster_data['price_sensitivity_index'].mean(),
                    'avg_colour_entropy': cluster_data['colour_preference_entropy'].mean(),
                    'avg_style_consistency': cluster_data['style_consistency_score'].mean()
                })
                
            if approach == 'behavioural':
                profile.update({
                    'avg_age': cluster_data['age'].mean(),
                    'active_pct': cluster_data['Active'].mean() * 100,
                    'club_member_pct': (cluster_data['club_member_status'] == 'ACTIVE').mean() * 100
                })
            
            cluster_profiles.append(profile)
        
        profiles_df = pd.DataFrame(cluster_profiles)
        
        # Generate business interpretations
        profiles_df['business_interpretation'] = self._generate_business_labels(profiles_df, approach)
        
        print(f"\nCluster Analysis Summary for {approach.title()} Approach:")
        print("="*60)
        for _, row in profiles_df.iterrows():
            print(f"Cluster {row['cluster']}: {row['business_interpretation']}")
            print(f"  Size: {row['size']:,} customers ({row['percentage']:.1f}%)")
            if approach == 'rfm' or 'avg_recency' in row:
                print(f"  RFM: R={row['avg_recency']:.1f}, F={row['avg_frequency']:.1f}, M=£{row['avg_monetary']:.2f}")
            print()
            
        return profiles_df
    
    def _generate_business_labels(self, profiles_df: pd.DataFrame, approach: str) -> List[str]:
        """Generate business-meaningful labels for clusters."""
        labels = []
        
        for _, row in profiles_df.iterrows():
            if approach == 'rfm':
                # RFM-based labels
                r, f, m = row['avg_recency'], row['avg_frequency'], row['avg_monetary']
                
                if r <= 30 and f >= 10 and m >= 500:
                    labels.append("Champions (High Value, High Engagement)")
                elif r <= 30 and f >= 5:
                    labels.append("Loyal Customers (Recent & Frequent)")
                elif m >= 300 and f >= 5:
                    labels.append("Big Spenders (High Value)")
                elif r >= 60 and f <= 3:
                    labels.append("At Risk (Declining Engagement)")
                elif r >= 70:
                    labels.append("Lost Customers (Dormant)")
                else:
                    labels.append("Developing Customers (Moderate Activity)")
                    
            elif approach == 'preference':
                # Preference-only based labels
                diversity = row.get('avg_diversity_score', 0)
                price_sens = row.get('avg_price_sensitivity', 0)
                colour_entropy = row.get('avg_colour_entropy', 0)
                style_consistency = row.get('avg_style_consistency', 0)
                
                if diversity >= 2.0 and colour_entropy >= 2.5 and style_consistency <= 0.3:
                    labels.append("Fashion Explorers (High Variety, Low Consistency)")
                elif diversity >= 1.5 and price_sens <= 0.8 and style_consistency >= 0.7:
                    labels.append("Premium Style Loyalists")
                elif price_sens >= 1.5 and diversity <= 1.0:
                    labels.append("Value-Focused Minimalists")
                elif colour_entropy >= 3.0 and diversity >= 1.2:
                    labels.append("Colour & Style Adventurers")
                elif style_consistency >= 0.8 and diversity <= 1.5:
                    labels.append("Consistent Style Followers")
                elif price_sens >= 1.2 and colour_entropy <= 2.0:
                    labels.append("Budget-Conscious Traditionalists")
                else:
                    labels.append("Moderate Preference Shoppers")
                    
            elif approach == 'hybrid':
                # RFM + Preferences based labels
                r, f, m = row['avg_recency'], row['avg_frequency'], row['avg_monetary']
                diversity = row.get('avg_diversity_score', 0)
                price_sens = row.get('avg_price_sensitivity', 0)
                
                if f >= 8 and m >= 400 and diversity >= 1.5:
                    labels.append("Premium Fashion Enthusiasts")
                elif f >= 5 and diversity >= 1.2 and price_sens <= 0.8:
                    labels.append("Diverse Style Seekers") 
                elif m >= 200 and price_sens >= 1.2:
                    labels.append("Budget-Conscious Shoppers")
                elif r <= 30 and f >= 3:
                    labels.append("Regular Customers")
                elif r >= 60:
                    labels.append("Inactive Customers")
                else:
                    labels.append("Casual Shoppers")
                    
            else:  # behavioural
                # Full feature behavioural labels
                f, m = row['avg_frequency'], row['avg_monetary']
                age = row.get('avg_age', 40)
                active_pct = row.get('active_pct', 50)
                
                if f >= 10 and m >= 500 and age <= 35:
                    labels.append("Young Fashion Leaders")
                elif f >= 8 and m >= 300 and active_pct >= 80:
                    labels.append("Highly Engaged Customers")
                elif age >= 50 and m >= 200:
                    labels.append("Mature Valuable Customers")
                elif f <= 2 and m <= 100:
                    labels.append("Minimal Engagement")
                else:
                    labels.append("Standard Customers")
                    
        return labels
    
    def visualise_clusters(self, approach: str = 'hybrid', method: str = 'kmeans', 
                          save_path: Optional[str] = None):
        """
        Create comprehensive cluster visualisations.
        
        Args:
            approach: Clustering approach to visualise
            method: Clustering method used
            save_path: Optional path to save plots
        """
        key = f'{approach}_{method}'
        
        if key not in self.cluster_results:
            raise ValueError(f"No clustering results found for {key}. Run perform_clustering first.")
            
        result_data = self.cluster_results[key]['data']
        cluster_col = self.cluster_results[key]['cluster_column']
        feature_cols = self.processed_data[approach]['feature_cols']
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Customer Clustering Analysis - {approach.title()} Approach', fontsize=16)
        
        # 1. Cluster size distribution
        cluster_sizes = result_data[cluster_col].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # 2. PCA visualization (2D)
        X = result_data[feature_cols].values
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=result_data[cluster_col], 
                                   cmap='tab10', alpha=0.6)
        axes[0, 1].set_title(f'PCA Visualisation (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Feature importance heatmap (cluster centers for K-means)
        if method == 'kmeans':
            model = self.models[key]
            cluster_centers = pd.DataFrame(model.cluster_centers_, 
                                         columns=feature_cols,
                                         index=[f'Cluster {i}' for i in range(len(model.cluster_centers_))])
            
            sns.heatmap(cluster_centers.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Cluster Centers (Standardised Features)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Cluster centers not available\nfor hierarchical clustering', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Cluster Analysis')
        
        # 4. RFM analysis (if available)
        if all(col in result_data.columns for col in ['customer_id']):
            # Merge with original data for RFM analysis
            original_data = self.df.to_pandas()
            rfm_data = result_data[['customer_id', cluster_col]].merge(
                original_data[['customer_id', 'recency', 'frequency', 'monetary']], 
                on='customer_id'
            )
            
            # Box plot of monetary value by cluster
            rfm_data.boxplot(column='monetary', by=cluster_col, ax=axes[1, 1])
            axes[1, 1].set_title('Monetary Value Distribution by Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Monetary Value (£)')
            plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation saved to {save_path}")
        
        plt.show()
    
    def export_results(self, approach: str = 'hybrid', method: str = 'kmeans', 
                      output_path: str = None) -> str:
        """
        Export clustering results to Parquet file.
        
        Args:
            approach: Clustering approach to export
            method: Clustering method used
            output_path: Path for output file
            
        Returns:
            Path to exported file
        """
        key = f'{approach}_{method}'
        
        if key not in self.cluster_results:
            raise ValueError(f"No clustering results found for {key}. Run perform_clustering first.")
            
        result_data = self.cluster_results[key]['data']
        cluster_col = self.cluster_results[key]['cluster_column']
        
        # Merge with original data
        original_data = self.df.to_pandas()
        export_data = original_data.merge(
            result_data[['customer_id', cluster_col]], 
            on='customer_id', how='left'
        )
        
        # Generate output path
        if output_path is None:
            output_path = f'data/results/customer_clusters_{approach}_{method}.parquet'
            
        # Export to Parquet
        export_df = pl.from_pandas(export_data)
        export_df.write_parquet(output_path)
        
        print(f"Clustering results exported to {output_path}")
        print(f"Exported {len(export_data):,} customers with {approach} clusters")
        
        return output_path


def main():
    """Example usage of CustomerSegmentation class."""
    
    # Initialise with sample for demonstration
    segmentation = CustomerSegmentation(
        'data/features/final/customers_features_final.parquet',
        sample_size=50000  # Use subset for faster processing
    )
    
    # 1. RFM-based clustering (most business interpretable)
    print("=== RFM-Based Clustering ===")
    optimal_rfm = segmentation.find_optimal_clusters('rfm', max_k=8)
    clusters_rfm = segmentation.perform_clustering('rfm', n_clusters=5)
    profiles_rfm = segmentation.analyze_clusters('rfm')
    
    # 2. Hybrid approach (recommended balance)
    print("\n=== Hybrid Clustering (RFM + Preferences) ===")
    optimal_hybrid = segmentation.find_optimal_clusters('hybrid', max_k=8)
    clusters_hybrid = segmentation.perform_clustering('hybrid', n_clusters=6)
    profiles_hybrid = segmentation.analyze_clusters('hybrid')
    
    # 3. Visualise results
    segmentation.visualise_clusters('hybrid')
    
    # 4. Export results
    export_path = segmentation.export_results('hybrid')
    
    print(f"\n=== Clustering Complete ===")
    print("Recommended approach: Hybrid clustering provides the best balance of")
    print("business interpretability and customer insight depth.")


if __name__ == "__main__":
    main()