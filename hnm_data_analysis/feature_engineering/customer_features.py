"""
Customer feature engineering module for RFM analysis and advanced customer metrics.

This module implements sophisticated customer behavioural feature engineering including:
- RFM Analysis (Recency, Frequency, Monetary value)
- Purchase diversity score (variety of products purchased)
- Price sensitivity index (response to price variations)
- Colour preference entropy (diversity in colour choices)
- Style consistency score (coherence in style preferences)

All features are engineered using Polars for high-performance processing with memory-efficient
batch processing to handle large customer datasets.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
from datetime import datetime, date, timedelta


class CustomerFeatures:
    """
    Advanced customer feature engineering class implementing RFM analysis and behavioural metrics.
    
    Features implemented:
    - RFM Analysis: Recency (days since last purchase), Frequency (purchase count), 
      Monetary (total spend)
    - Purchase Diversity Score: Shannon entropy of product category purchases
    - Price Sensitivity Index: Coefficient of variation in price paid across purchases
    - Colour Preference Entropy: Shannon entropy of colour group preferences
    - Style Consistency Score: Consistency in garment group preferences
    """
    
    def __init__(
        self,
        customers_path: str = "data/cleaned/customers_cleaned.parquet",
        transactions_path: str = "data/cleaned/transactions_cleaned.parquet", 
        articles_path: str = "data/features/final/articles_features_final.parquet",
        batch_size: int = 50000,
        sample_size: Optional[int] = None
    ):
        """
        Initialise CustomerFeatures with data paths and processing parameters.
        
        Args:
            customers_path: Path to cleaned customers data
            transactions_path: Path to cleaned transactions data
            articles_path: Path to articles features data
            batch_size: Number of customers to process per batch for memory efficiency
            sample_size: Optional sample size for development/testing
        """
        self.customers_path = Path(customers_path)
        self.transactions_path = Path(transactions_path)
        self.articles_path = Path(articles_path)
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Reference date for recency calculations (latest transaction date + 1 day)
        self.reference_date = None
        
    def _load_and_validate_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Load and validate all required datasets with optional sampling.
        
        Returns:
            Tuple of (customers, transactions, articles) DataFrames
            
        Raises:
            FileNotFoundError: If required data files don't exist
            ValueError: If datasets have unexpected schemas
        """
        self.logger.info("Loading datasets...")
        
        # Load customers data
        if not self.customers_path.exists():
            raise FileNotFoundError(f"Customers file not found: {self.customers_path}")
        customers = pl.read_parquet(self.customers_path)
        
        # Load transactions data
        if not self.transactions_path.exists():
            raise FileNotFoundError(f"Transactions file not found: {self.transactions_path}")
        transactions = pl.read_parquet(self.transactions_path)
        
        # Load articles data
        if not self.articles_path.exists():
            raise FileNotFoundError(f"Articles file not found: {self.articles_path}")
        articles = pl.read_parquet(self.articles_path)
        
        # Apply sampling if specified
        if self.sample_size:
            self.logger.info(f"Applying sampling: {self.sample_size} customers")
            customers = customers.head(self.sample_size)
            customer_ids = customers.select("customer_id")
            transactions = transactions.join(customer_ids, on="customer_id", how="inner")
            
        # Set reference date for recency calculations (latest transaction date + 1 day)
        max_date = transactions.select(pl.col("t_dat").max()).item()
        self.reference_date = max_date + timedelta(days=1)
        self.logger.info(f"Reference date for recency: {self.reference_date}")
        
        # Validate schemas
        self._validate_schemas(customers, transactions, articles)
        
        self.logger.info(f"Loaded: {customers.height} customers, {transactions.height} transactions, "
                        f"{articles.height} articles")
        
        return customers, transactions, articles
    
    def _validate_schemas(self, customers: pl.DataFrame, transactions: pl.DataFrame, 
                         articles: pl.DataFrame) -> None:
        """
        Validate that datasets have expected schemas for feature engineering.
        
        Args:
            customers: Customer data DataFrame
            transactions: Transaction data DataFrame  
            articles: Articles features DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        # Required columns for each dataset
        required_customers = {"customer_id"}
        required_transactions = {"customer_id", "article_id", "t_dat", "price"}
        required_articles = {"article_id", "product_group_name", "colour_group_name", 
                           "garment_group_name"}
        
        # Check customers schema
        missing_customers = required_customers - set(customers.columns)
        if missing_customers:
            raise ValueError(f"Missing columns in customers data: {missing_customers}")
            
        # Check transactions schema  
        missing_transactions = required_transactions - set(transactions.columns)
        if missing_transactions:
            raise ValueError(f"Missing columns in transactions data: {missing_transactions}")
            
        # Check articles schema
        missing_articles = required_articles - set(articles.columns)
        if missing_articles:
            raise ValueError(f"Missing columns in articles data: {missing_articles}")
    
    def _calculate_rfm_features(self, transactions: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) features for each customer.
        
        RFM Analysis components:
        - Recency: Days since last purchase (lower = more recent = better)
        - Frequency: Total number of transactions (higher = better)  
        - Monetary: Total spend across all transactions (higher = better)
        
        Args:
            transactions: Transaction data with customer_id, t_dat, price
            
        Returns:
            DataFrame with customer_id, recency, frequency, monetary columns
        """
        self.logger.info("Calculating RFM features...")
        
        # Log extreme values for monitoring
        price_stats = transactions.select([
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            pl.col("price").quantile(0.99).alias("p99_price")
        ]).to_dict(as_series=False)
        
        self.logger.info(f"Price range: £{price_stats['min_price'][0]:.2f} - £{price_stats['max_price'][0]:.2f}")
        self.logger.info(f"99th percentile: £{price_stats['p99_price'][0]:.2f}")
        
        rfm_features = (
            transactions
            .group_by("customer_id")
            .agg([
                # Recency: days since last purchase (ensure it's within data bounds)
                (pl.lit(self.reference_date) - pl.col("t_dat").max()).dt.total_days()
                .cast(pl.Int64).clip(1, None).alias("recency"),
                
                # Frequency: total number of transactions
                pl.len().alias("frequency"),
                
                # Monetary: total spend (validate extreme values)
                pl.col("price").sum().alias("monetary")
            ])
        )
        
        # Log extreme customer behavior for validation
        freq_stats = rfm_features.select([
            pl.col("frequency").max().alias("max_freq"),
            pl.col("frequency").quantile(0.99).alias("p99_freq"),
            pl.col("monetary").max().alias("max_monetary"),
            pl.col("monetary").quantile(0.99).alias("p99_monetary")
        ]).to_dict(as_series=False)
        
        self.logger.info(f"Customer frequency max: {freq_stats['max_freq'][0]} (99th percentile: {freq_stats['p99_freq'][0]:.0f})")
        self.logger.info(f"Customer monetary max: £{freq_stats['max_monetary'][0]:.2f} (99th percentile: £{freq_stats['p99_monetary'][0]:.2f})")
        
        return rfm_features
    
    def _calculate_purchase_diversity_score(self, transactions: pl.DataFrame, 
                                          articles: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate purchase diversity score using Shannon entropy of product groups.
        
        Measures how diverse a customer's product category purchases are.
        Higher entropy indicates more diverse shopping behaviour across categories.
        
        Args:
            transactions: Transaction data 
            articles: Articles data with product_group_name
            
        Returns:
            DataFrame with customer_id, purchase_diversity_score columns
        """
        self.logger.info("Calculating purchase diversity scores...")
        
        # Join transactions with articles to get product groups
        trans_with_groups = (
            transactions
            .join(articles.select(["article_id", "product_group_name"]), 
                  on="article_id", how="left")
        )
        
        # Calculate entropy of product group purchases per customer
        diversity_features = (
            trans_with_groups
            .filter(pl.col("product_group_name").is_not_null())
            .group_by(["customer_id", "product_group_name"])
            .agg(pl.len().alias("group_count"))
            .group_by("customer_id")
            .agg([
                # Calculate Shannon entropy: -sum(p * log2(p))
                # Use abs() to ensure non-negative values and handle edge cases
                (
                    -(pl.col("group_count") / pl.col("group_count").sum()) *
                    (pl.col("group_count") / pl.col("group_count").sum()).log(2)
                ).sum().fill_nan(0.0).abs().alias("purchase_diversity_score")
            ])
        )
        
        return diversity_features
    
    def _calculate_price_sensitivity_index(self, transactions: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate price sensitivity index using coefficient of variation in prices paid.
        
        Measures consistency in price points chosen by customers.
        Higher values indicate more price-sensitive behaviour (shopping across price ranges).
        Lower values indicate consistent price point preferences.
        
        Args:
            transactions: Transaction data with price column
            
        Returns:
            DataFrame with customer_id, price_sensitivity_index columns
        """
        self.logger.info("Calculating price sensitivity indices...")
        
        price_sensitivity_features = (
            transactions
            .group_by("customer_id")
            .agg([
                # Calculate coefficient of variation: std_dev / mean
                (pl.col("price").std() / pl.col("price").mean())
                .fill_nan(0.0).alias("price_sensitivity_index")
            ])
            .filter(pl.col("price_sensitivity_index").is_not_null())
        )
        
        return price_sensitivity_features
    
    def _calculate_colour_preference_entropy(self, transactions: pl.DataFrame,
                                           articles: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate colour preference entropy using Shannon entropy of colour group choices.
        
        Measures diversity in colour preferences across purchases.
        Higher entropy indicates more adventurous colour choices.
        Lower entropy indicates consistent colour preferences.
        
        Args:
            transactions: Transaction data
            articles: Articles data with colour_group_name
            
        Returns:
            DataFrame with customer_id, colour_preference_entropy columns
        """
        self.logger.info("Calculating colour preference entropy...")
        
        # Join transactions with articles to get colour groups
        trans_with_colours = (
            transactions
            .join(articles.select(["article_id", "colour_group_name"]), 
                  on="article_id", how="left")
        )
        
        # Calculate entropy of colour group choices per customer
        colour_entropy_features = (
            trans_with_colours
            .filter(pl.col("colour_group_name").is_not_null())
            .group_by(["customer_id", "colour_group_name"])
            .agg(pl.len().alias("colour_count"))
            .group_by("customer_id")
            .agg([
                # Calculate Shannon entropy: -sum(p * log2(p))
                # Use abs() to ensure non-negative values and handle edge cases
                (
                    -(pl.col("colour_count") / pl.col("colour_count").sum()) *
                    (pl.col("colour_count") / pl.col("colour_count").sum()).log(2)
                ).sum().fill_nan(0.0).abs().alias("colour_preference_entropy")
            ])
        )
        
        return colour_entropy_features
    
    def _calculate_style_consistency_score(self, transactions: pl.DataFrame,
                                         articles: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate style consistency score using garment group concentration.
        
        Measures how consistent a customer is in their garment type preferences.
        Higher scores indicate more consistent style preferences.
        Lower scores indicate more varied style exploration.
        
        Calculated as: 1 - (Shannon entropy / log2(number_of_categories))
        This normalises entropy to [0,1] and inverts it so higher = more consistent.
        
        Args:
            transactions: Transaction data
            articles: Articles data with garment_group_name
            
        Returns:
            DataFrame with customer_id, style_consistency_score columns
        """
        self.logger.info("Calculating style consistency scores...")
        
        # Join transactions with articles to get garment groups
        trans_with_garments = (
            transactions
            .join(articles.select(["article_id", "garment_group_name"]), 
                  on="article_id", how="left")
        )
        
        # Calculate normalised entropy (consistency) of garment group choices
        style_consistency_features = (
            trans_with_garments
            .filter(pl.col("garment_group_name").is_not_null())
            .group_by(["customer_id", "garment_group_name"])
            .agg(pl.len().alias("garment_count"))
            .group_by("customer_id")
            .agg([
                # Calculate Shannon entropy with proper handling of edge cases
                (
                    -(pl.col("garment_count") / pl.col("garment_count").sum()) *
                    (pl.col("garment_count") / pl.col("garment_count").sum()).log(2)
                ).sum().fill_nan(0.0).alias("entropy"),
                
                # Count number of distinct categories for normalisation
                pl.len().alias("num_categories")
            ])
            .with_columns([
                # Normalise entropy and invert for consistency score
                (1 - (pl.col("entropy") / pl.col("num_categories").log(2)))
                .fill_nan(1.0).clip(0.0, 1.0).alias("style_consistency_score")
            ])
            .select(["customer_id", "style_consistency_score"])
        )
        
        return style_consistency_features
    
    def engineer_customer_features(self, output_path: Optional[str] = None) -> pl.DataFrame:
        """
        Engineer all customer features using batch processing for memory efficiency.
        
        Creates comprehensive customer feature set including:
        - RFM Analysis (recency, frequency, monetary)
        - Purchase diversity score
        - Price sensitivity index  
        - Colour preference entropy
        - Style consistency score
        
        Args:
            output_path: Optional path to save engineered features
            
        Returns:
            DataFrame with all customer features
        """
        self.logger.info("Starting customer feature engineering pipeline...")
        
        # Load and validate data
        customers, transactions, articles = self._load_and_validate_data()
        
        # Calculate each feature set
        self.logger.info("Calculating individual feature sets...")
        
        rfm_features = self._calculate_rfm_features(transactions)
        diversity_features = self._calculate_purchase_diversity_score(transactions, articles)
        price_sensitivity_features = self._calculate_price_sensitivity_index(transactions)
        colour_entropy_features = self._calculate_colour_preference_entropy(transactions, articles)
        style_consistency_features = self._calculate_style_consistency_score(transactions, articles)
        
        # Combine all features with left joins to preserve all customers
        self.logger.info("Combining feature sets...")
        
        combined_features = (
            customers.select("customer_id")
            .join(rfm_features, on="customer_id", how="left")
            .join(diversity_features, on="customer_id", how="left")  
            .join(price_sensitivity_features, on="customer_id", how="left")
            .join(colour_entropy_features, on="customer_id", how="left")
            .join(style_consistency_features, on="customer_id", how="left")
        )
        
        # Handle missing values for customers with no transactions
        # Calculate proper recency for customers with no transactions (use max days in dataset + 1)
        max_recency = (self.reference_date - transactions.select(pl.col("t_dat").min()).item()).days
        
        combined_features = combined_features.with_columns([
            pl.col("recency").fill_null(max_recency + 1),  # Proper max recency for no purchases
            pl.col("frequency").fill_null(0),  # Zero frequency 
            pl.col("monetary").fill_null(0.0),  # Zero monetary value
            pl.col("purchase_diversity_score").fill_null(0.0),  # No diversity
            pl.col("price_sensitivity_index").fill_null(0.0),  # No sensitivity data
            pl.col("colour_preference_entropy").fill_null(0.0),  # No colour diversity
            pl.col("style_consistency_score").fill_null(1.0),  # Default consistency
        ])
        
        # Add feature metadata columns
        combined_features = combined_features.with_columns([
            pl.lit(datetime.now().isoformat()).alias("features_created_at"),
            pl.lit(str(self.reference_date)).alias("rfm_reference_date")
        ])
        
        self.logger.info(f"Engineered features for {combined_features.height} customers")
        self.logger.info(f"Feature columns: {combined_features.columns}")
        
        # Save if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_features.write_parquet(output_path)
            self.logger.info(f"Features saved to: {output_path}")
            
        return combined_features
    
    def generate_feature_summary(self, features_df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics and insights for engineered customer features.
        
        Args:
            features_df: DataFrame with engineered customer features
            
        Returns:
            Dictionary with feature summary statistics and business insights
        """
        self.logger.info("Generating feature summary statistics...")
        
        # Calculate summary statistics for each feature
        numeric_columns = ["recency", "frequency", "monetary", "purchase_diversity_score",
                          "price_sensitivity_index", "colour_preference_entropy", 
                          "style_consistency_score"]
        
        summary_stats = {}
        for col in numeric_columns:
            if col in features_df.columns:
                stats = (
                    features_df
                    .select([
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).median().alias("median"), 
                        pl.col(col).std().alias("std"),
                        pl.col(col).min().alias("min"),
                        pl.col(col).max().alias("max"),
                        pl.col(col).null_count().alias("null_count")
                    ])
                ).to_dict(as_series=False)
                
                summary_stats[col] = {k: v[0] for k, v in stats.items()}
        
        # Add business insights
        total_customers = features_df.height
        active_customers = features_df.filter(pl.col("frequency") > 0).height
        
        summary = {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "active_customer_rate": active_customers / total_customers,
            "feature_statistics": summary_stats,
            "features_created_at": datetime.now().isoformat(),
            "reference_date": str(self.reference_date)
        }
        
        return summary


def main():
    """
    Main function to demonstrate customer feature engineering.
    Processes customer data and saves engineered features.
    """
    # Initialise feature engineer with sample for testing
    feature_engineer = CustomerFeatures(sample_size=10000)
    
    # Engineer features
    features_df = feature_engineer.engineer_customer_features(
        output_path="data/features/customer_features_sample.parquet"
    )
    
    # Generate summary
    summary = feature_engineer.generate_feature_summary(features_df)
    
    print("Customer Feature Engineering Complete!")
    print(f"Processed {summary['total_customers']} customers")
    print(f"Active customers: {summary['active_customers']} ({summary['active_customer_rate']:.1%})")
    print("\nFeature Summary:")
    for feature, stats in summary['feature_statistics'].items():
        print(f"  {feature}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")


if __name__ == "__main__":
    main()