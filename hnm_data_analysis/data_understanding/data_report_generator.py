"""
Data Report Generator Module

This module provides functionality to generate Markdown reports
for data files (CSV and Parquet) to help understand dataset characteristics,
quality, and structure.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime


class DataReportGenerator:
    """
    Generates data understanding reports for CSV and Parquet files.
    
    The report includes:
    - Dataset overview (shape, file info)
    - Schema information (column types, nullability)
    - Data quality assessment (missing values, duplicates)
    - Statistical summaries
    - Data type analysis
    - Memory usage information
    """
    
    def __init__(self, file_path: str, sample_size: Optional[int] = None):
        """
        Initialise the data report generator.
        
        Args:
            file_path: Path to the CSV or Parquet file
            sample_size: Optional sample size for large datasets (default: full dataset)
        """
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.df = None
        self.report_sections = []
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.file_path.suffix.lower() not in ['.csv', '.parquet']:
            raise ValueError("File must be either CSV or Parquet format")
    
    def load_data(self) -> pl.DataFrame:
        """Load data from file with optional sampling.

        For CSV files, when a `sample_size` is provided we only read the first N rows
        using Polars' `n_rows` parameter to avoid loading the entire dataset into memory.
        """
        try:
            if self.file_path.suffix.lower() == '.csv':
                if self.sample_size:
                    # Read only the first N rows to keep memory usage reasonable on huge CSVs
                    self.df = pl.read_csv(self.file_path, n_rows=self.sample_size)
                else:
                    self.df = pl.read_csv(self.file_path)
            else:  # parquet
                if self.sample_size:
                    # Parquet supports efficient row selection but Polars does not expose n_rows for parquet
                    # so we read fully and then take the head N rows which is typically cheap for parquet
                    self.df = pl.read_parquet(self.file_path).head(self.sample_size)
                else:
                    self.df = pl.read_parquet(self.file_path)
            
            return self.df
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def _is_probable_id_column(self, column_name: str) -> bool:
        """
        Heuristic to detect identifier-like columns to exclude from numeric summaries.
        A column is considered an ID if its name suggests identifiers or it has a
        very high uniqueness ratio relative to the number of rows.
        """
        name = column_name.lower()
        name_is_id_like = (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or name.endswith("id")
            or "uuid" in name
            or "guid" in name
        )
        total_rows = len(self.df) if self.df is not None else 0
        try:
            uniqueness_ratio = (self.df[column_name].n_unique() / total_rows) if total_rows > 0 else 0.0
        except Exception:
            uniqueness_ratio = 0.0
        is_highly_unique = uniqueness_ratio >= 0.95
        return name_is_id_like or is_highly_unique

    def _find_project_root(self) -> Path:
        """
        Determine the repository/project root reliably across OS and execution contexts.
        Preference order:
        - Directory containing common repo markers: `.git`, `requirements.txt`, or `README.md`
        - The parent directories of this module (`__file__`)
        - Fallback to current working directory
        """
        # Start from the directory containing this module, not from the data file path
        start_dir = Path(__file__).resolve().parent
        repo_markers = {".git", "requirements.txt", "README.md"}
        for parent in [start_dir, *start_dir.parents]:
            try:
                entries = {p.name for p in parent.iterdir()}
            except Exception:
                continue
            if repo_markers.intersection(entries):
                return parent
        # Fallback: two levels up from this module (expected repo root structure)
        try:
            return start_dir.parents[1]
        except Exception:
            return Path.cwd().resolve()
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get basic file information."""
        file_stats = self.file_path.stat()
        return {
            'file_name': self.file_path.name,
            'file_path': str(self.file_path),
            'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
            'file_type': self.file_path.suffix.upper().replace('.', ''),
            'last_modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyse_schema(self) -> Dict[str, Any]:
        """Analyse dataset schema and structure."""
        schema_info = {
            'total_columns': len(self.df.columns),
            'total_rows': len(self.df),
            'column_details': []
        }
        
        for col_name in self.df.columns:
            col_info = {
                'name': col_name,
                'dtype': str(self.df[col_name].dtype),
                'null_count': self.df[col_name].null_count(),
                'null_percentage': round((self.df[col_name].null_count() / len(self.df)) * 100, 2),
                'unique_count': self.df[col_name].n_unique(),
                'unique_percentage': round((self.df[col_name].n_unique() / len(self.df)) * 100, 2)
            }
            schema_info['column_details'].append(col_info)
        
        return schema_info
    
    def analyse_data_quality(self) -> Dict[str, Any]:
        """Analyse data quality metrics."""
        total_cells = len(self.df) * len(self.df.columns)
        total_nulls = sum(self.df[col].null_count() for col in self.df.columns)
        
        # Check for duplicate rows
        duplicate_count = len(self.df) - self.df.unique().height
        
        quality_metrics = {
            'total_cells': total_cells,
            'total_nulls': total_nulls,
            'null_percentage': round((total_nulls / total_cells) * 100, 2),
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': round((duplicate_count / len(self.df)) * 100, 2),
            'completeness_score': round(((total_cells - total_nulls) / total_cells) * 100, 2)
        }
        
        return quality_metrics
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary for numeric columns."""
        numeric_cols = [
            col
            for col in self.df.columns
            if self.df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        ]
        # Exclude probable identifier columns from numeric summaries
        summarizable_cols = [col for col in numeric_cols if not self._is_probable_id_column(col)]
        
        if not summarizable_cols:
            return {'message': 'No numeric columns found for statistical analysis'}
        
        stats_summary = {}
        for col in summarizable_cols:
            try:
                # Use individual methods instead of describe() to avoid access issues
                col_series = self.df[col]
                
                # Calculate statistics using individual methods
                count_val = col_series.count()
                mean_val = col_series.mean()
                std_val = col_series.std()
                min_val = col_series.min()
                max_val = col_series.max()
                
                # Calculate quantiles
                quantiles = col_series.quantile([0.25, 0.50, 0.75])
                q25 = quantiles[0] if len(quantiles) > 0 else None
                q50 = quantiles[1] if len(quantiles) > 1 else None
                q75 = quantiles[2] if len(quantiles) > 2 else None
                
                stats_summary[col] = {
                    'count': count_val,
                    'mean': round(mean_val, 4) if mean_val is not None else None,
                    'std': round(std_val, 4) if std_val is not None else None,
                    'min': min_val,
                    'max': max_val,
                    '25%': q25,
                    '50%': q50,
                    '75%': q75
                }
            except Exception as e:
                # Fallback for columns that don't support the operations above
                stats_summary[col] = {
                    'count': self.df[col].count() if hasattr(self.df[col], 'count') else None,
                    'min': self.df[col].min() if hasattr(self.df[col], 'min') else None,
                    'max': self.df[col].max() if hasattr(self.df[col], 'max') else None,
                    'mean': round(self.df[col].mean(), 4) if hasattr(self.df[col], 'mean') and self.df[col].mean() is not None else None,
                    'std': None,
                    '25%': None,
                    '50%': None,
                    '75%': None
                }
        
        return stats_summary
    
    def analyse_categorical_columns(self) -> Dict[str, Any]:
        """Analyse categorical/string columns."""
        categorical_cols = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]
        
        if not categorical_cols:
            return {'message': 'No categorical columns found'}
        
        categorical_analysis = {}
        for col in categorical_cols:
            try:
                value_counts = self.df[col].value_counts().head(10)
                categorical_analysis[col] = {
                    'unique_count': self.df[col].n_unique(),
                    'most_frequent_values': value_counts.to_dicts()[:5] if len(value_counts) > 0 else [],
                    'avg_length': round(self.df[col].str.len_chars().mean(), 2) if self.df[col].dtype == pl.Utf8 else None
                }
            except Exception:
                categorical_analysis[col] = {
                    'unique_count': self.df[col].n_unique(),
                    'most_frequent_values': [],
                    'avg_length': None
                }
        
        return categorical_analysis
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage information."""
        try:
            # Get estimated memory usage per column
            memory_info = {
                'estimated_memory_mb': round(self.df.estimated_size("mb"), 2),
                'memory_per_column': {}
            }
            
            for col in self.df.columns:
                col_memory = self.df.select(col).estimated_size("mb")
                memory_info['memory_per_column'][col] = round(col_memory, 4)
            
            return memory_info
        except Exception:
            return {'message': 'Memory usage estimation not available'}
    
    def generate_markdown_report(self) -> str:
        """Generate Markdown report for data understanding."""
        if self.df is None:
            self.load_data()
        
        # Collect all analysis data
        file_info = self.get_file_info()
        schema_info = self.analyse_schema()
        quality_metrics = self.analyse_data_quality()
        statistical_summary = self.get_statistical_summary()
        categorical_analysis = self.analyse_categorical_columns()
        memory_info = self.get_memory_usage()
        
        # Build Markdown report
        report = []
        
        # Title and overview
        report.append(f"# Data Understanding Report")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**File:** {file_info['file_name']}")
        if self.sample_size:
            report.append(f"**Note:** Analysis based on sample of {self.sample_size:,} records")
        report.append("")
        
        # File Information
        report.append("## 📄 File Information")
        report.append(f"- **File Path:** `{file_info['file_path']}`")
        report.append(f"- **File Type:** {file_info['file_type']}")
        report.append(f"- **File Size:** {file_info['file_size_mb']} MB")
        report.append(f"- **Last Modified:** {file_info['last_modified']}")
        report.append("")
        
        # Dataset Overview
        report.append("## 📊 Dataset Overview")
        report.append(f"- **Rows:** {schema_info['total_rows']:,}")
        report.append(f"- **Columns:** {schema_info['total_columns']}")
        report.append(f"- **Total Cells:** {quality_metrics['total_cells']:,}")
        report.append("")
        
        # Data Quality Summary
        report.append("## 🔍 Data Quality Summary")
        report.append(f"- **Completeness Score:** {quality_metrics['completeness_score']}%")
        report.append(f"- **Missing Values:** {quality_metrics['total_nulls']:,} ({quality_metrics['null_percentage']}%)")
        report.append(f"- **Duplicate Rows:** {quality_metrics['duplicate_rows']:,} ({quality_metrics['duplicate_percentage']}%)")
        report.append("")
        
        # Schema Information
        report.append("## 📋 Schema Information")
        report.append("| Column | Data Type | Null Count | Null % | Unique Count | Unique % |")
        report.append("|--------|-----------|------------|---------|--------------|----------|")
        
        for col in schema_info['column_details']:
            report.append(f"| {col['name']} | {col['dtype']} | {col['null_count']:,} | {col['null_percentage']}% | {col['unique_count']:,} | {col['unique_percentage']}% |")
        report.append("")
        
        # Statistical Summary
        if 'message' not in statistical_summary:
            report.append("## 📈 Statistical Summary (Numeric Columns)")
            report.append("| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |")
            report.append("|--------|-------|------|-----|-----|-----|-----|-----|-----|")
            
            for col, stats in statistical_summary.items():
                mean_val = f"{stats['mean']}" if stats['mean'] is not None else "N/A"
                std_val = f"{stats['std']}" if stats['std'] is not None else "N/A"
                min_val = f"{stats['min']}" if stats['min'] is not None else "N/A"
                max_val = f"{stats['max']}" if stats['max'] is not None else "N/A"
                q25_val = f"{stats['25%']}" if stats['25%'] is not None else "N/A"
                q50_val = f"{stats['50%']}" if stats['50%'] is not None else "N/A"
                q75_val = f"{stats['75%']}" if stats['75%'] is not None else "N/A"
                count_val = f"{stats['count']}" if stats['count'] is not None else "N/A"
                
                report.append(f"| {col} | {count_val} | {mean_val} | {std_val} | {min_val} | {q25_val} | {q50_val} | {q75_val} | {max_val} |")
            report.append("")
        
        # Categorical Analysis
        if 'message' not in categorical_analysis:
            report.append("## 📝 Categorical Column Analysis")
            for col, analysis in categorical_analysis.items():
                report.append(f"### {col}")
                report.append(f"- **Unique Values:** {analysis['unique_count']:,}")
                if analysis['avg_length']:
                    report.append(f"- **Average Length:** {analysis['avg_length']} characters")
                
                if analysis['most_frequent_values']:
                    report.append("- **Top Values:**")
                    for value_info in analysis['most_frequent_values']:
                        value = value_info.get(col, 'N/A')
                        count = value_info.get('count', 0)
                        report.append(f"  - `{value}`: {count:,} occurrences")
                report.append("")
        
        # Memory Usage
        if 'message' not in memory_info:
            report.append("## 💾 Memory Usage")
            report.append(f"- **Estimated Total Memory:** {memory_info['estimated_memory_mb']} MB")
            report.append("")
            report.append("### Memory by Column")
            report.append("| Column | Memory (MB) |")
            report.append("|--------|-------------|")
            for col, memory in memory_info['memory_per_column'].items():
                report.append(f"| {col} | {memory} |")
            report.append("")
        
        # Data Quality Recommendations
        report.append("## 💡 Data Quality Recommendations")
        recommendations = []
        
        if quality_metrics['null_percentage'] > 10:
            recommendations.append("- **High missing values detected** - Consider imputation strategies or investigate data collection issues")
        
        if quality_metrics['duplicate_percentage'] > 5:
            recommendations.append("- **Significant duplicates found** - Review and consider deduplication")
        
        # Check for potential ID columns
        id_columns = [col['name'] for col in schema_info['column_details'] if col['unique_percentage'] > 95]
        if id_columns:
            recommendations.append(f"- **Potential ID columns detected:** {', '.join(id_columns)} - Verify if these should be used as identifiers")
        
        # Check for low cardinality columns
        low_cardinality = [col['name'] for col in schema_info['column_details'] if col['unique_count'] <= 10 and col['unique_count'] > 1]
        if low_cardinality:
            recommendations.append(f"- **Low cardinality columns:** {', '.join(low_cardinality)} - Good candidates for categorical encoding")
        
        if not recommendations:
            recommendations.append("- **No major data quality issues detected**")
        
        for rec in recommendations:
            report.append(rec)
        
        report.append("")
        report.append("---")
        report.append("*Report generated using HnM Data Analytics - Data Understanding Module*")
        
        return "\n".join(report)
    
    def save_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate and save the Markdown report.
        
        Args:
            output_path: Optional custom output path. If not provided, 
                        saves to results/data_documentation/data_reports/ directory
        
        Returns:
            Path to the saved report file
        """
        if output_path is None:
            # Default to repository-level results/data_documentation/data_reports directory
            project_root = self._find_project_root()
            results_dir = project_root / "results" / "data_documentation" / "data_reports"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / f"{self.file_path.stem}_data_report.md"
        else:
            output_path = Path(output_path)
        
        report_content = self.generate_markdown_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Print confirmation message
        print(f"Data report generated successfully!")
        print(f"Analysed file: {self.file_path}")
        print(f"Report saved to: {output_path}")
        
        # Return relative path from project root
        try:
            relative_path = output_path.relative_to(project_root)
            return str(relative_path)
        except (ValueError, NameError):
            # Fallback if project_root not found
            return str(output_path)


def generate_data_report(file_path: str, output_path: Optional[str] = None, sample_size: Optional[int] = None) -> str:
    """
    Convenience function to generate a data understanding report.
    
    Args:
        file_path: Path to the CSV or Parquet file
        output_path: Optional output path for the report
        sample_size: Optional sample size for large datasets
    
    Returns:
        Path to the generated report file
    """
    generator = DataReportGenerator(file_path, sample_size)
    return generator.save_report(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_report_generator.py <file_path> [output_path] [sample_size]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    try:
        report_path = generate_data_report(file_path, output_path, sample_size)
        print(f"Data report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)