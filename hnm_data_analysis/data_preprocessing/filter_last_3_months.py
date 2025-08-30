"""
Filter H&M transaction data to keep only the final 3 months of transactions.

This script loads the transactions_train.csv file and filters it to only include
transactions from the last 3 months of the dataset (2020-06-23 to 2020-09-22).
Uses Polars for efficient processing of the large dataset.
"""

import polars as pl
from datetime import datetime, timedelta
import os


class TransactionFilter:
    """
    A class to filter H&M transaction data to keep only the final 3 months of transactions.
    
    This class provides methods to load, filter, and save transaction data using Polars
    for efficient processing of large datasets.
    """
    
    def __init__(self, input_path: str):
        """
        Initialise the TransactionFilter with the input data path.
        
        Args:
            input_path: Path to the input transactions_train.csv file
        """
        self.input_path = input_path
        self.original_df = None
        self.filtered_df = None
        self.max_date = None
        self.three_months_prior = None
        
    def load_data(self) -> None:
        """Load the transaction data from the input path."""
        print("Loading transaction data...")
        self.original_df = pl.read_csv(self.input_path)
        
        print(f"Original dataset shape: {self.original_df.shape}")
        print(f"Date range: {self.original_df['t_dat'].min()} to {self.original_df['t_dat'].max()}")
        
        # Convert t_dat to date type if it's not already
        self.original_df = self.original_df.with_columns(
            pl.col("t_dat").str.to_date().alias("t_dat")
        )
        
    def calculate_date_range(self) -> None:
        """Calculate the date range for filtering (last 3 months)."""
        if self.original_df is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
            
        self.max_date = self.original_df['t_dat'].max()
        self.three_months_prior = self.max_date - timedelta(days=90)  # Approximately 3 months
        
        print(f"Filtering for dates from {self.three_months_prior} to {self.max_date}")
        
    def filter_transactions(self) -> pl.DataFrame:
        """
        Filter transactions to keep only the final 3 months.
        
        Returns:
            Polars DataFrame with filtered transactions
        """
        if self.original_df is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
            
        if self.three_months_prior is None:
            self.calculate_date_range()
            
        # Filter for the last 3 months
        self.filtered_df = self.original_df.filter(pl.col("t_dat") >= self.three_months_prior)
        
        print(f"Filtered dataset shape: {self.filtered_df.shape}")
        print(f"Percentage of original data retained: {(self.filtered_df.shape[0] / self.original_df.shape[0]) * 100:.2f}%")
        
        return self.filtered_df
        
    def save_data(self, output_csv_path: str = None, output_parquet_path: str = None) -> None:
        """
        Save the filtered data to specified paths.
        
        Args:
            output_csv_path: Optional path to save the filtered data as CSV
            output_parquet_path: Optional path to save the filtered data as Parquet
        """
        if self.filtered_df is None:
            raise ValueError("Data must be filtered first. Call filter_transactions() method.")
            
        # Save to output paths if provided
        if output_csv_path:
            print(f"Saving filtered data as CSV to {output_csv_path}")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            self.filtered_df.write_csv(output_csv_path)
        
        if output_parquet_path:
            print(f"Saving filtered data as Parquet to {output_parquet_path}")
            os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
            self.filtered_df.write_parquet(output_parquet_path)
            
    def get_statistics(self) -> dict:
        """
        Get basic statistics about the filtered data.
        
        Returns:
            Dictionary containing basic statistics
        """
        if self.filtered_df is None:
            raise ValueError("Data must be filtered first. Call filter_transactions() method.")
            
        return {
            'num_transactions': self.filtered_df.shape[0],
            'num_unique_customers': self.filtered_df['customer_id'].n_unique(),
            'num_unique_articles': self.filtered_df['article_id'].n_unique(),
            'date_range_start': self.filtered_df['t_dat'].min(),
            'date_range_end': self.filtered_df['t_dat'].max()
        }
        
    def process(self, output_csv_path: str = None, output_parquet_path: str = None) -> pl.DataFrame:
        """
        Complete processing pipeline: load, filter, and optionally save data.
        
        Args:
            output_csv_path: Optional path to save the filtered data as CSV
            output_parquet_path: Optional path to save the filtered data as Parquet
            
        Returns:
            Polars DataFrame with filtered transactions
        """
        self.load_data()
        filtered_df = self.filter_transactions()
        
        if output_csv_path or output_parquet_path:
            self.save_data(output_csv_path, output_parquet_path)
            
        return filtered_df


def main():
    """Main function to execute the filtering."""
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_path = os.path.join(base_path, "data", "raw", "transactions_train.csv")
    output_csv_path = os.path.join(base_path, "data", "processed", "transactions_last_3_months.csv")
    output_parquet_path = os.path.join(base_path, "data", "processed", "transactions_last_3_months.parquet")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    try:
        # Create and use the TransactionFilter class
        transaction_filter = TransactionFilter(input_path)
        filtered_data = transaction_filter.process(output_csv_path, output_parquet_path)
        
        print("\nFiltering completed successfully!")
        print(f"Filtered data saved to:")
        print(f"  CSV: {output_csv_path}")
        print(f"  Parquet: {output_parquet_path}")
        
        # Report file sizes
        print(f"\nFile sizes:")
        if os.path.exists(output_csv_path):
            csv_size = os.path.getsize(output_csv_path)
            print(f"  CSV: {csv_size:,} bytes ({csv_size / (1024**2):.2f} MB)")
        
        if os.path.exists(output_parquet_path):
            parquet_size = os.path.getsize(output_parquet_path)
            print(f"  Parquet: {parquet_size:,} bytes ({parquet_size / (1024**2):.2f} MB)")
            
            if os.path.exists(output_csv_path):
                compression_ratio = (csv_size - parquet_size) / csv_size * 100
                print(f"  Compression: Parquet is {compression_ratio:.1f}% smaller than CSV")
        
        # Display statistics using the class method
        print("\nBasic statistics of filtered data:")
        stats = transaction_filter.get_statistics()
        print(f"Number of transactions: {stats['num_transactions']:,}")
        print(f"Number of unique customers: {stats['num_unique_customers']:,}")
        print(f"Number of unique articles: {stats['num_unique_articles']:,}")
        print(f"Date range: {stats['date_range_start']} to {stats['date_range_end']}")
        
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()