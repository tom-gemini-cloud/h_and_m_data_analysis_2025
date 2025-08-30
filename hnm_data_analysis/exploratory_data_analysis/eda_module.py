"""
Comprehensive EDA module for H&M data analysis with high-quality visualisations for Word documents.

This module provides exploratory data analysis capabilities for any dataset with:
- Data profiling and quality assessment
- Distribution analysis and visualisation
- Correlation analysis with heatmaps
- Customer segmentation insights
- Time series pattern analysis
- A4 Word document-optimised plotting

All visualisations are designed to be publication-ready for business reports.
"""

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import warnings
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EDAModule:
    """
    Comprehensive Exploratory Data Analysis module for customer and retail data.
    
    Features:
    - Automatic data profiling and quality assessment
    - Distribution analysis with statistical tests
    - Correlation analysis and visualisation
    - Customer segmentation using clustering
    - Time series pattern detection
    - Publication-quality plots for Word documents
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (10, 6),
        dpi: int = 300,
        style: str = 'whitegrid',
        colour_palette: str = 'viridis'
    ):
        """
        Initialise EDA module with plotting configuration for Word documents.
        
        Args:
            figure_size: Default figure size in inches (width, height)
            dpi: Dots per inch for high-quality output
            style: Seaborn style theme
            colour_palette: Default colour palette for plots
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.style = style
        self.colour_palette = colour_palette
        
        # Configure plotting for Word document compatibility
        self._setup_plotting_style()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
    
    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn for professional A4 Word document output."""
        plt.style.use('seaborn-v0_8')
        sns.set_style(self.style)
        sns.set_palette(self.colour_palette)
        
        # Configure matplotlib for high-quality output
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'DejaVu Sans'
        })
    
    def load_data(self, file_path: str, **kwargs) -> pl.DataFrame:
        """
        Load data from various file formats using Polars for efficiency.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Polars DataFrame with loaded data
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        if path.suffix.lower() == '.parquet':
            df = pl.read_parquet(file_path, **kwargs)
        elif path.suffix.lower() == '.csv':
            df = pl.read_csv(file_path, **kwargs)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            # Convert to pandas first, then to polars
            pd_df = pd.read_excel(file_path, **kwargs)
            df = pl.from_pandas(pd_df)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self.logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    def generate_data_profile(self, df: pl.DataFrame, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profiling report.
        
        Args:
            df: Input DataFrame
            save_path: Optional path to save profile report
            
        Returns:
            Dictionary containing profiling results
        """
        self.logger.info("Generating data profile...")
        
        # Basic dataset information
        profile = {
            'dataset_shape': df.shape,
            'memory_usage_mb': df.estimated_size('mb'),
            'column_types': dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),
            'missing_values': {},
            'duplicate_rows': df.height - df.n_unique(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Missing values analysis
        for col in df.columns:
            null_count = df.select(pl.col(col).null_count()).item()
            null_percentage = (null_count / df.height) * 100
            profile['missing_values'][col] = {
                'count': null_count,
                'percentage': null_percentage
            }
        
        # Numeric columns summary
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        for col in numeric_cols:
            if col in df.columns:
                stats_data = df.select([
                    pl.col(col).mean().alias('mean'),
                    pl.col(col).median().alias('median'),
                    pl.col(col).std().alias('std'),
                    pl.col(col).min().alias('min'),
                    pl.col(col).max().alias('max'),
                    pl.col(col).quantile(0.25).alias('q25'),
                    pl.col(col).quantile(0.75).alias('q75')
                ]).to_dict(as_series=False)
                
                profile['numeric_summary'][col] = {k: v[0] for k, v in stats_data.items()}
        
        # Categorical columns summary
        categorical_cols = df.select(pl.col(pl.Utf8)).columns
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df.select(pl.col(col).n_unique()).item()
                most_frequent = df.group_by(col).count().sort('count', descending=True).head(1)
                
                profile['categorical_summary'][col] = {
                    'unique_count': unique_count,
                    'most_frequent_value': most_frequent.select(col).item() if most_frequent.height > 0 else None,
                    'most_frequent_count': most_frequent.select('count').item() if most_frequent.height > 0 else None
                }
        
        # Save profile if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            self.logger.info(f"Data profile saved to: {save_path}")
        
        return profile
    
    def plot_distributions(
        self, 
        df: pl.DataFrame, 
        columns: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> List[plt.Figure]:
        """
        Create distribution plots for numeric columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to plot (defaults to all numeric)
            save_dir: Directory to save plots
            
        Returns:
            List of matplotlib figures
        """
        if columns is None:
            columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        self.logger.info(f"Creating distribution plots for {len(columns)} columns...")
        
        figures = []
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Distribution Analysis: {col}', fontsize=16, fontweight='bold')
            
            # Convert to pandas for plotting
            data = df.select(col).to_pandas()[col].dropna()
            
            # Histogram
            axes[0, 0].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Histogram')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[0, 1].boxplot(data, vert=True)
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_ylabel(col)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Density plot
            data.plot.density(ax=axes[1, 1], title='Density Plot')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures.append(fig)
            
            # Save if directory specified
            if save_dir:
                save_path = Path(save_dir) / f'{col}_distribution.png'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                
        return figures
    
    def plot_correlation_matrix(
        self, 
        df: pl.DataFrame, 
        method: str = 'pearson',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create correlation heatmap for numeric columns.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        self.logger.info(f"Creating correlation matrix using {method} method...")
        
        # Get numeric columns only
        numeric_df = df.select(pl.col(pl.NUMERIC_DTYPES))
        
        if numeric_df.width == 0:
            raise ValueError("No numeric columns found for correlation analysis")
        
        # Convert to pandas for correlation calculation
        corr_matrix = numeric_df.to_pandas().corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title(f'Correlation Matrix ({method.title()})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if path specified
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Correlation matrix saved to: {save_path}")
        
        return fig
    
    def customer_segmentation_analysis(
        self,
        df: pl.DataFrame,
        features: List[str],
        n_clusters: int = 5,
        save_dir: Optional[str] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any], List[plt.Figure]]:
        """
        Perform customer segmentation using K-means clustering.
        
        Args:
            df: Input DataFrame
            features: List of feature columns for clustering
            n_clusters: Number of clusters
            save_dir: Directory to save analysis results
            
        Returns:
            Tuple of (dataframe with clusters, cluster analysis, figures)
        """
        self.logger.info(f"Performing customer segmentation with {n_clusters} clusters...")
        
        # Prepare data for clustering
        cluster_data = df.select(features).to_pandas().fillna(0)
        
        # Standardise features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add clusters to dataframe
        df_with_clusters = df.with_columns(pl.Series('cluster', cluster_labels))
        
        # Cluster analysis
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            cluster_data_subset = cluster_data[cluster_mask]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(cluster_labels) * 100),
                'means': cluster_data_subset.mean().to_dict(),
                'medians': cluster_data_subset.median().to_dict()
            }
        
        figures = []
        
        # Create cluster visualisation using PCA
        if len(features) > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_title('Customer Segments (PCA Projection)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            figures.append(fig)
        
        # Create cluster summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Cluster Analysis Summary', fontsize=16, fontweight='bold')
        
        # Cluster sizes
        cluster_sizes = [cluster_analysis[f'cluster_{i}']['size'] for i in range(n_clusters)]
        axes[0, 0].pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(n_clusters)], autopct='%1.1f%%')
        axes[0, 0].set_title('Cluster Size Distribution')
        
        # Feature importance (first few features)
        for idx, feature in enumerate(features[:3]):
            if idx < 3:
                means = [cluster_analysis[f'cluster_{i}']['means'][feature] for i in range(n_clusters)]
                axes[0, 1].bar([f'C{i}' for i in range(n_clusters)], means, alpha=0.7, label=feature)
        axes[0, 1].set_title('Mean Feature Values by Cluster')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        figures.append(fig)
        
        # Save results if directory specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save cluster assignments
            df_with_clusters.write_csv(save_path / 'customer_segments.csv')
            
            # Save cluster analysis
            import json
            with open(save_path / 'cluster_analysis.json', 'w') as f:
                json.dump(cluster_analysis, f, indent=2, default=str)
            
            # Save figures
            for i, fig in enumerate(figures):
                fig.savefig(save_path / f'cluster_analysis_{i}.png', dpi=self.dpi, bbox_inches='tight')
        
        return df_with_clusters, cluster_analysis, figures
    
    def time_series_analysis(
        self,
        df: pl.DataFrame,
        date_column: str,
        value_column: str,
        freq: str = 'D',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Perform time series analysis and visualisation.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column to analyse
            freq: Frequency for aggregation ('D', 'W', 'M')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        self.logger.info(f"Performing time series analysis on {value_column}...")
        
        # Prepare time series data
        ts_data = (
            df.select([date_column, value_column])
            .with_columns(pl.col(date_column).cast(pl.Date))
            .group_by(date_column)
            .agg(pl.col(value_column).sum())
            .sort(date_column)
        )
        
        # Convert to pandas for time series operations
        ts_pandas = ts_data.to_pandas().set_index(date_column)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Time Series Analysis: {value_column}', fontsize=16, fontweight='bold')
        
        # Main time series plot
        axes[0].plot(ts_pandas.index, ts_pandas[value_column], linewidth=1.5)
        axes[0].set_title('Time Series')
        axes[0].set_ylabel(value_column)
        axes[0].grid(True, alpha=0.3)
        
        # Rolling statistics
        window = min(30, len(ts_pandas) // 4)
        if window > 1:
            rolling_mean = ts_pandas[value_column].rolling(window=window).mean()
            rolling_std = ts_pandas[value_column].rolling(window=window).std()
            
            axes[1].plot(ts_pandas.index, rolling_mean, label=f'{window}-day Moving Average', linewidth=2)
            axes[1].fill_between(ts_pandas.index, 
                               rolling_mean - rolling_std, 
                               rolling_mean + rolling_std, 
                               alpha=0.3, label='±1 Std Dev')
            axes[1].set_title('Rolling Statistics')
            axes[1].set_ylabel(value_column)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Distribution of values
        axes[2].hist(ts_pandas[value_column], bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_title('Distribution of Daily Values')
        axes[2].set_xlabel(value_column)
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path specified
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Time series analysis saved to: {save_path}")
        
        return fig
    
    def comprehensive_eda_report(
        self,
        df: pl.DataFrame,
        target_column: Optional[str] = None,
        output_dir: str = "eda_results"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report with all analyses and visualisations.
        
        Args:
            df: Input DataFrame
            target_column: Optional target variable for supervised learning context
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting comprehensive EDA analysis...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data profile
        profile = self.generate_data_profile(df, str(output_path / 'data_profile.json'))
        
        # Create distribution plots
        dist_figures = self.plot_distributions(df, save_dir=str(output_path / 'distributions'))
        
        # Create correlation matrix
        corr_fig = self.plot_correlation_matrix(df, save_path=str(output_path / 'correlation_matrix.png'))
        
        # Customer segmentation (if enough numeric features)
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        segmentation_results = None
        if len(numeric_cols) >= 3:
            # Use top numeric features for segmentation
            top_features = numeric_cols[:min(6, len(numeric_cols))]
            df_clustered, cluster_analysis, cluster_figs = self.customer_segmentation_analysis(
                df, top_features, save_dir=str(output_path / 'segmentation')
            )
            segmentation_results = {
                'data': df_clustered,
                'analysis': cluster_analysis,
                'figures': cluster_figs
            }
        
        # Time series analysis (if date column exists)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        ts_results = None
        if date_columns and len(numeric_cols) > 0:
            date_col = date_columns[0]
            value_col = numeric_cols[0]
            ts_fig = self.time_series_analysis(
                df, date_col, value_col, save_path=str(output_path / 'time_series_analysis.png')
            )
            ts_results = {'figure': ts_fig}
        
        # Compile comprehensive report
        report = {
            'dataset_profile': profile,
            'distribution_analysis': {
                'figures': dist_figures,
                'summary': f"Generated {len(dist_figures)} distribution plots"
            },
            'correlation_analysis': {
                'figure': corr_fig,
                'summary': f"Correlation matrix for {len(numeric_cols)} numeric features"
            },
            'segmentation_analysis': segmentation_results,
            'time_series_analysis': ts_results,
            'report_generated_at': datetime.now().isoformat(),
            'output_directory': str(output_path)
        }
        
        self.logger.info(f"Comprehensive EDA report completed. Results saved to: {output_path}")
        
        return report


def main():
    """
    Example usage of the EDA module with H&M customer data.
    """
    # Initialise EDA module
    eda = EDAModule()
    
    # Load sample data (replace with actual data path)
    try:
        df = eda.load_data("data/features/final/customer_features_final.parquet")
        
        # Generate comprehensive EDA report
        report = eda.comprehensive_eda_report(df, output_dir="results/eda_analysis")
        
        print("EDA Analysis Complete!")
        print(f"Dataset shape: {report['dataset_profile']['dataset_shape']}")
        print(f"Results saved to: {report['output_directory']}")
        
    except FileNotFoundError:
        print("Sample data not found. Please provide a valid data file path.")
        print("Example usage:")
        print("eda = EDAModule()")
        print("df = eda.load_data('your_data_file.parquet')")
        print("report = eda.comprehensive_eda_report(df)")


if __name__ == "__main__":
    main()