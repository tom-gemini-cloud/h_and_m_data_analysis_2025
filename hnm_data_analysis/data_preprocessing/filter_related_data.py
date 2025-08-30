"""
Filter H&M articles and customers data to keep only those present in the last 3 months transactions.

This script loads the filtered transactions from the last 3 months and then filters
the articles.csv and customers.csv files to only include records that appear in
those transactions. Uses Polars for efficient processing.
"""

import polars as pl
import os


class DataFilter:
    """
    A class to filter H&M articles and customers data based on transaction data.
    
    This class provides methods to load transaction data and filter articles and customers
    datasets to keep only records that appear in the transactions.
    """
    
    def __init__(self, transactions_path: str, articles_path: str, customers_path: str):
        """
        Initialise the DataFilter with input data paths.
        
        Args:
            transactions_path: Path to the filtered transactions file (last 3 months)
            articles_path: Path to the articles.csv file
            customers_path: Path to the customers.csv file
        """
        self.transactions_path = transactions_path
        self.articles_path = articles_path
        self.customers_path = customers_path
        
        self.transactions_df = None
        self.articles_df = None
        self.customers_df = None
        self.filtered_articles = None
        self.filtered_customers = None
        self.unique_article_ids = None
        self.unique_customer_ids = None
        
    def load_transactions(self) -> None:
        """Load the transaction data (try parquet first, then CSV)."""
        print("Loading transaction data (last 3 months)...")
        
        if self.transactions_path.endswith('.parquet'):
            self.transactions_df = pl.read_parquet(self.transactions_path)
        else:
            self.transactions_df = pl.read_csv(self.transactions_path)
        
        print(f"Transactions shape: {self.transactions_df.shape}")
        
    def extract_unique_ids(self) -> None:
        """Extract unique article IDs and customer IDs from transactions."""
        if self.transactions_df is None:
            raise ValueError("Transactions data must be loaded first. Call load_transactions() method.")
            
        self.unique_article_ids = self.transactions_df.select("article_id").unique()
        self.unique_customer_ids = self.transactions_df.select("customer_id").unique()
        
        print(f"Unique articles in transactions: {self.unique_article_ids.shape[0]:,}")
        print(f"Unique customers in transactions: {self.unique_customer_ids.shape[0]:,}")
        
    def load_and_filter_articles(self) -> pl.DataFrame:
        """
        Load and filter articles data based on unique article IDs from transactions.
        
        Returns:
            Filtered articles DataFrame
        """
        if self.unique_article_ids is None:
            raise ValueError("Unique IDs must be extracted first. Call extract_unique_ids() method.")
            
        print("\nLoading and filtering articles data...")
        self.articles_df = pl.read_csv(self.articles_path)
        print(f"Original articles shape: {self.articles_df.shape}")
        
        self.filtered_articles = self.articles_df.join(
            self.unique_article_ids,
            on="article_id",
            how="inner"
        )
        print(f"Filtered articles shape: {self.filtered_articles.shape}")
        print(f"Articles retained: {(self.filtered_articles.shape[0] / self.articles_df.shape[0]) * 100:.2f}%")
        
        return self.filtered_articles
        
    def load_and_filter_customers(self) -> pl.DataFrame:
        """
        Load and filter customers data based on unique customer IDs from transactions.
        
        Returns:
            Filtered customers DataFrame
        """
        if self.unique_customer_ids is None:
            raise ValueError("Unique IDs must be extracted first. Call extract_unique_ids() method.")
            
        print("\nLoading and filtering customers data...")
        self.customers_df = pl.read_csv(self.customers_path)
        print(f"Original customers shape: {self.customers_df.shape}")
        
        self.filtered_customers = self.customers_df.join(
            self.unique_customer_ids,
            on="customer_id",
            how="inner"
        )
        print(f"Filtered customers shape: {self.filtered_customers.shape}")
        print(f"Customers retained: {(self.filtered_customers.shape[0] / self.customers_df.shape[0]) * 100:.2f}%")
        
        return self.filtered_customers
        
    def save_articles(self, output_csv_path: str = None, output_parquet_path: str = None) -> None:
        """
        Save filtered articles data to specified paths.
        
        Args:
            output_csv_path: Optional path to save filtered articles as CSV
            output_parquet_path: Optional path to save filtered articles as Parquet
        """
        if self.filtered_articles is None:
            raise ValueError("Articles must be filtered first. Call load_and_filter_articles() method.")
            
        if output_csv_path:
            print(f"\nSaving filtered articles as CSV to {output_csv_path}")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            self.filtered_articles.write_csv(output_csv_path)
        
        if output_parquet_path:
            print(f"Saving filtered articles as Parquet to {output_parquet_path}")
            os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
            self.filtered_articles.write_parquet(output_parquet_path)
            
    def save_customers(self, output_csv_path: str = None, output_parquet_path: str = None) -> None:
        """
        Save filtered customers data to specified paths.
        
        Args:
            output_csv_path: Optional path to save filtered customers as CSV
            output_parquet_path: Optional path to save filtered customers as Parquet
        """
        if self.filtered_customers is None:
            raise ValueError("Customers must be filtered first. Call load_and_filter_customers() method.")
            
        if output_csv_path:
            print(f"Saving filtered customers as CSV to {output_csv_path}")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            self.filtered_customers.write_csv(output_csv_path)
        
        if output_parquet_path:
            print(f"Saving filtered customers as Parquet to {output_parquet_path}")
            os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
            self.filtered_customers.write_parquet(output_parquet_path)
            
    def get_statistics(self) -> dict:
        """
        Get statistics about the filtering results.
        
        Returns:
            Dictionary containing filtering statistics
        """
        if self.filtered_articles is None or self.filtered_customers is None:
            raise ValueError("Both articles and customers must be filtered first.")
            
        return {
            'original_articles_count': self.articles_df.shape[0],
            'filtered_articles_count': self.filtered_articles.shape[0],
            'articles_retention_rate': (self.filtered_articles.shape[0] / self.articles_df.shape[0]) * 100,
            'original_customers_count': self.customers_df.shape[0],
            'filtered_customers_count': self.filtered_customers.shape[0], 
            'customers_retention_rate': (self.filtered_customers.shape[0] / self.customers_df.shape[0]) * 100,
            'unique_articles_in_transactions': self.unique_article_ids.shape[0],
            'unique_customers_in_transactions': self.unique_customer_ids.shape[0]
        }
        
    def process(self, 
                output_articles_csv: str = None,
                output_articles_parquet: str = None,
                output_customers_csv: str = None,
                output_customers_parquet: str = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Complete processing pipeline: load, filter, and optionally save data.
        
        Args:
            output_articles_csv: Optional path to save filtered articles as CSV
            output_articles_parquet: Optional path to save filtered articles as Parquet
            output_customers_csv: Optional path to save filtered customers as CSV
            output_customers_parquet: Optional path to save filtered customers as Parquet
            
        Returns:
            Tuple of (filtered_articles_df, filtered_customers_df)
        """
        self.load_transactions()
        self.extract_unique_ids()
        
        filtered_articles = self.load_and_filter_articles()
        filtered_customers = self.load_and_filter_customers()
        
        # Save if paths provided
        if output_articles_csv or output_articles_parquet:
            self.save_articles(output_articles_csv, output_articles_parquet)
            
        if output_customers_csv or output_customers_parquet:
            self.save_customers(output_customers_csv, output_customers_parquet)
            
        return filtered_articles, filtered_customers


def report_file_sizes(file_paths: list[str], labels: list[str]):
    """Report file sizes for given files."""
    print(f"\nFile sizes:")
    for path, label in zip(file_paths, labels):
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {label}: {size:,} bytes ({size / (1024**2):.2f} MB)")


def main():
    """Main function to execute the filtering."""
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Input paths
    transactions_path = os.path.join(base_path, "data", "processed", "transactions_last_3_months.parquet")
    # Fallback to CSV if parquet doesn't exist
    if not os.path.exists(transactions_path):
        transactions_path = os.path.join(base_path, "data", "processed", "transactions_last_3_months.csv")
    
    articles_path = os.path.join(base_path, "data", "raw", "articles.csv")
    customers_path = os.path.join(base_path, "data", "raw", "customers.csv")
    
    # Output paths
    output_articles_csv = os.path.join(base_path, "data", "processed", "articles_last_3_months.csv")
    output_articles_parquet = os.path.join(base_path, "data", "processed", "articles_last_3_months.parquet")
    output_customers_csv = os.path.join(base_path, "data", "processed", "customers_last_3_months.csv")
    output_customers_parquet = os.path.join(base_path, "data", "processed", "customers_last_3_months.parquet")
    
    # Check if required input files exist
    required_files = [transactions_path, articles_path, customers_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Required input files not found:")
        for f in missing_files:
            print(f"  {f}")
        print("\nPlease run the transaction filtering script first or ensure all data files are present.")
        return
    
    try:
        # Create and use the DataFilter class
        data_filter = DataFilter(transactions_path, articles_path, customers_path)
        filtered_articles, filtered_customers = data_filter.process(
            output_articles_csv=output_articles_csv,
            output_articles_parquet=output_articles_parquet,
            output_customers_csv=output_customers_csv,
            output_customers_parquet=output_customers_parquet
        )
        
        print("\n" + "="*60)
        print("FILTERING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nFiltered data saved to:")
        print(f"Articles:")
        print(f"  CSV: {output_articles_csv}")
        print(f"  Parquet: {output_articles_parquet}")
        print(f"Customers:")
        print(f"  CSV: {output_customers_csv}")
        print(f"  Parquet: {output_customers_parquet}")
        
        # Report file sizes
        file_paths = [
            output_articles_csv, output_articles_parquet,
            output_customers_csv, output_customers_parquet
        ]
        labels = [
            "Articles CSV", "Articles Parquet",
            "Customers CSV", "Customers Parquet"
        ]
        report_file_sizes(file_paths, labels)
        
        # Calculate compression ratios
        print(f"\nCompression ratios:")
        if os.path.exists(output_articles_csv) and os.path.exists(output_articles_parquet):
            articles_csv_size = os.path.getsize(output_articles_csv)
            articles_parquet_size = os.path.getsize(output_articles_parquet)
            articles_compression = (articles_csv_size - articles_parquet_size) / articles_csv_size * 100
            print(f"  Articles: Parquet is {articles_compression:.1f}% smaller than CSV")
        
        if os.path.exists(output_customers_csv) and os.path.exists(output_customers_parquet):
            customers_csv_size = os.path.getsize(output_customers_csv)
            customers_parquet_size = os.path.getsize(output_customers_parquet)
            customers_compression = (customers_csv_size - customers_parquet_size) / customers_csv_size * 100
            print(f"  Customers: Parquet is {customers_compression:.1f}% smaller than CSV")
        
        # Display statistics using the class method
        print(f"\nFinal Summary:")
        stats = data_filter.get_statistics()
        print(f"  Filtered articles: {stats['filtered_articles_count']:,} records")
        print(f"  Filtered customers: {stats['filtered_customers_count']:,} records")
        print(f"  Article retention rate: {stats['articles_retention_rate']:.2f}%")
        print(f"  Customer retention rate: {stats['customers_retention_rate']:.2f}%")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()