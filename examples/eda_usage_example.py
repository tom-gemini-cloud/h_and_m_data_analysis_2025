"""
Example usage of the EDA module for H&M data analysis.

This script demonstrates how to use the EDA module to perform comprehensive
exploratory data analysis on customer, transaction, and article datasets.
"""

from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hnm_data_analysis.exploratory_data_analysis.eda_module import EDAModule


def analyse_customer_features():
    """Example: Comprehensive EDA on customer features dataset."""
    print("🔍 Customer Features EDA Analysis")
    print("=" * 50)
    
    # Initialise EDA module with A4 Word document settings
    eda = EDAModule(
        figure_size=(10, 6),
        dpi=300,
        style='whitegrid',
        colour_palette='viridis'
    )
    
    try:
        # Load customer features data
        data_path = "data/features/final/customer_features_final.parquet"
        df = eda.load_data(data_path)
        
        print(f"✅ Loaded dataset: {df.shape[0]:,} customers × {df.shape[1]} features")
        
        # Generate comprehensive EDA report
        report = eda.comprehensive_eda_report(
            df, 
            output_dir="results/eda_analysis/customer_features"
        )
        
        # Print summary
        print("\n📊 Analysis Summary:")
        print(f"• Dataset shape: {report['dataset_profile']['dataset_shape']}")
        print(f"• Memory usage: {report['dataset_profile']['memory_usage_mb']:.1f} MB")
        print(f"• Missing values: {sum(1 for col_info in report['dataset_profile']['missing_values'].values() if col_info['count'] > 0)} columns")
        print(f"• Numeric features: {len(report['dataset_profile']['numeric_summary'])}")
        print(f"• Results saved to: {report['output_directory']}")
        
        return report
        
    except FileNotFoundError:
        print("❌ Customer features data not found. Please ensure the data exists.")
        return None


def analyse_transaction_patterns():
    """Example: Time series analysis on transaction data."""
    print("\n🕒 Transaction Patterns Analysis")
    print("=" * 50)
    
    eda = EDAModule()
    
    try:
        # Load transaction data
        data_path = "data/cleaned/transactions_cleaned.parquet"
        df = eda.load_data(data_path)
        
        print(f"✅ Loaded dataset: {df.shape[0]:,} transactions")
        
        # Perform time series analysis on transaction volume
        ts_fig = eda.time_series_analysis(
            df,
            date_column='t_dat',
            value_column='price',
            save_path="results/eda_analysis/transaction_patterns.png"
        )
        
        print("📈 Time series analysis completed")
        print("• Generated transaction volume trends")
        print("• Analysed seasonal patterns")
        print("• Results saved to: results/eda_analysis/transaction_patterns.png")
        
        return ts_fig
        
    except FileNotFoundError:
        print("❌ Transaction data not found. Please ensure the data exists.")
        return None


def analyse_article_features():
    """Example: Distribution and correlation analysis on article features."""
    print("\n🏷️  Article Features Analysis")
    print("=" * 50)
    
    eda = EDAModule()
    
    try:
        # Load article features data
        data_path = "data/features/final/articles_features_final.parquet"
        df = eda.load_data(data_path)
        
        print(f"✅ Loaded dataset: {df.shape[0]:,} articles × {df.shape[1]} features")
        
        # Generate data profile
        profile = eda.generate_data_profile(
            df, 
            save_path="results/eda_analysis/article_features_profile.json"
        )
        
        # Create distribution plots for numeric features
        dist_figs = eda.plot_distributions(
            df, 
            save_dir="results/eda_analysis/article_distributions"
        )
        
        # Create correlation matrix
        corr_fig = eda.plot_correlation_matrix(
            df,
            save_path="results/eda_analysis/article_correlations.png"
        )
        
        print("📊 Analysis completed:")
        print(f"• Generated {len(dist_figs)} distribution plots")
        print("• Created correlation matrix")
        print("• Data profile saved")
        
        return profile, dist_figs, corr_fig
        
    except FileNotFoundError:
        print("❌ Article features data not found. Please ensure the data exists.")
        return None


def perform_customer_segmentation():
    """Example: Customer segmentation analysis using RFM features."""
    print("\n👥 Customer Segmentation Analysis")
    print("=" * 50)
    
    eda = EDAModule()
    
    try:
        # Load customer features data
        data_path = "data/features/final/customer_features_final.parquet"
        df = eda.load_data(data_path)
        
        # Define RFM features for segmentation
        rfm_features = ['recency', 'frequency', 'monetary']
        
        # Check if RFM features exist
        available_features = [f for f in rfm_features if f in df.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient RFM features for segmentation")
            return None
        
        # Perform customer segmentation
        df_clustered, cluster_analysis, cluster_figs = eda.customer_segmentation_analysis(
            df,
            features=available_features,
            n_clusters=5,
            save_dir="results/eda_analysis/customer_segmentation"
        )
        
        print("🎯 Segmentation completed:")
        print(f"• Identified 5 customer segments")
        print(f"• Used features: {', '.join(available_features)}")
        
        # Print cluster summary
        for cluster_id, info in cluster_analysis.items():
            print(f"• {cluster_id}: {info['size']:,} customers ({info['percentage']:.1f}%)")
        
        print("• Results saved to: results/eda_analysis/customer_segmentation/")
        
        return df_clustered, cluster_analysis, cluster_figs
        
    except FileNotFoundError:
        print("❌ Customer features data not found. Please ensure the data exists.")
        return None


def main():
    """
    Run comprehensive EDA examples on H&M datasets.
    """
    print("🎯 H&M Data Analysis - EDA Module Examples")
    print("=" * 60)
    
    # Create output directory
    output_path = Path("results/eda_analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run different EDA analyses
    customer_report = analyse_customer_features()
    transaction_fig = analyse_transaction_patterns()
    article_results = analyse_article_features()
    segmentation_results = perform_customer_segmentation()
    
    print("\n🎉 EDA Analysis Examples Completed!")
    print("=" * 60)
    
    successful_analyses = []
    if customer_report: successful_analyses.append("Customer Features")
    if transaction_fig: successful_analyses.append("Transaction Patterns")
    if article_results: successful_analyses.append("Article Features")
    if segmentation_results: successful_analyses.append("Customer Segmentation")
    
    if successful_analyses:
        print(f"✅ Successfully completed: {', '.join(successful_analyses)}")
        print(f"📁 All results saved to: {output_path}")
        print("\n💡 Tips for Word Documents:")
        print("• All plots are optimised for A4 page size")
        print("• High DPI (300) ensures crisp printing")
        print("• Professional colour schemes for presentations")
        print("• Clear labels and titles for business context")
    else:
        print("❌ No analyses could be completed. Please check data availability.")
        print("\n📋 Required data files:")
        print("• data/features/final/customer_features_final.parquet")
        print("• data/cleaned/transactions_cleaned.parquet")
        print("• data/features/final/articles_features_final.parquet")


if __name__ == "__main__":
    main()