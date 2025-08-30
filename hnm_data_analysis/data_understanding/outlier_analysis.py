"""
Outlier Analysis Module

This module provides comprehensive outlier detection and analysis functionality
for numerical fields in datasets. It implements multiple statistical methods
and generates detailed Markdown reports suitable for data science assignments.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json
from datetime import datetime
from dataclasses import dataclass


@dataclass
class OutlierSummary:
    """Summary statistics for outlier detection results."""
    column: str
    total_values: int
    outliers_iqr: int
    outliers_zscore: int
    outliers_modified_zscore: int
    outliers_percentile: int
    q1: float
    q3: float
    iqr: float
    lower_bound_iqr: float
    upper_bound_iqr: float
    mean: float
    std: float
    min_val: float
    max_val: float
    p01: float
    p99: float


class OutlierAnalyser:
    """
    Outlier detection and analysis for numerical datasets.
    
    This class implements multiple statistical methods for outlier detection:
    - Interquartile Range (IQR) method
    - Z-score method
    - Modified Z-score method (using median absolute deviation)
    - Percentile-based method
    
    The analyser generates detailed Markdown reports with statistical summaries,
    outlier counts, and recommendations for data cleaning.
    """
    
    def __init__(self, file_path: str, sample_size: Optional[int] = None):
        """
        Initialise the outlier analyser.
        
        Args:
            file_path: Path to the CSV or Parquet file
            sample_size: Optional sample size for large datasets (default: full dataset)
        """
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.df = None
        self.outlier_results = {}
        self.numerical_columns = []
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.file_path.suffix.lower() not in ['.csv', '.parquet']:
            raise ValueError("File must be either CSV or Parquet format")
    
    def load_data(self) -> pl.DataFrame:
        """Load data from file with optional sampling."""
        try:
            if self.file_path.suffix.lower() == '.csv':
                if self.sample_size:
                    # For large CSV files, read with streaming and sample
                    df = pl.read_csv(self.file_path, n_rows=self.sample_size)
                else:
                    df = pl.read_csv(self.file_path)
            else:  # parquet
                df = pl.read_parquet(self.file_path)
                if self.sample_size and len(df) > self.sample_size:
                    df = df.sample(n=self.sample_size, seed=42)
            
            self.df = df
            self.numerical_columns = [
                col for col in df.columns 
                if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
                and 'id' not in col.lower()
            ]
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> Tuple[pl.Expr, float, float, float, float]:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            column: Column name to analyse
            multiplier: IQR multiplier (default 1.5 for standard outlier detection)
            
        Returns:
            Tuple of (outlier_mask, Q1, Q3, IQR, lower_bound, upper_bound)
        """
        col_data = self.df[column]
        
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
        
        return outlier_mask, q1, q3, iqr, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> Tuple[pl.Expr, float, float]:
        """
        Detect outliers using the Z-score method.
        
        Args:
            column: Column name to analyse
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Tuple of (outlier_mask, mean, std)
        """
        col_data = self.df[column]
        mean = col_data.mean()
        std = col_data.std()
        
        if std == 0:
            # Handle constant columns
            outlier_mask = pl.lit(False)
        else:
            z_scores = (pl.col(column) - mean) / std
            outlier_mask = z_scores.abs() > threshold
        
        return outlier_mask, mean, std
    
    def detect_outliers_modified_zscore(self, column: str, threshold: float = 3.5) -> pl.Expr:
        """
        Detect outliers using the Modified Z-score method (using median absolute deviation).
        
        Args:
            column: Column name to analyse
            threshold: Modified Z-score threshold (default 3.5)
            
        Returns:
            Outlier mask expression
        """
        col_data = self.df[column]
        median = col_data.median()
        
        # Calculate median absolute deviation (MAD)
        mad = (col_data - median).abs().median()
        
        if mad == 0:
            # Handle constant columns or columns with many identical values
            outlier_mask = pl.lit(False)
        else:
            modified_z_scores = 0.6745 * (pl.col(column) - median) / mad
            outlier_mask = modified_z_scores.abs() > threshold
        
        return outlier_mask
    
    def detect_outliers_percentile(self, column: str, lower_percentile: float = 1.0, 
                                   upper_percentile: float = 99.0) -> Tuple[pl.Expr, float, float]:
        """
        Detect outliers using percentile-based method.
        
        Args:
            column: Column name to analyse
            lower_percentile: Lower percentile threshold (default 1.0)
            upper_percentile: Upper percentile threshold (default 99.0)
            
        Returns:
            Tuple of (outlier_mask, lower_bound, upper_bound)
        """
        col_data = self.df[column]
        
        lower_bound = col_data.quantile(lower_percentile / 100)
        upper_bound = col_data.quantile(upper_percentile / 100)
        
        outlier_mask = (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
        
        return outlier_mask, lower_bound, upper_bound
    
    def analyse_column_outliers(self, column: str) -> OutlierSummary:
        """
        Comprehensive outlier analysis for a single column.
        
        Args:
            column: Column name to analyse
            
        Returns:
            OutlierSummary with all outlier detection results
        """
        col_data = self.df[column]
        total_values = len(col_data)
        
        # IQR method
        iqr_mask, q1, q3, iqr, lower_iqr, upper_iqr = self.detect_outliers_iqr(column)
        outliers_iqr = self.df.filter(iqr_mask).height
        
        # Z-score method
        zscore_mask, mean_val, std_val = self.detect_outliers_zscore(column)
        outliers_zscore = self.df.filter(zscore_mask).height
        
        # Modified Z-score method
        mod_zscore_mask = self.detect_outliers_modified_zscore(column)
        outliers_modified_zscore = self.df.filter(mod_zscore_mask).height
        
        # Percentile method
        perc_mask, p01, p99 = self.detect_outliers_percentile(column)
        outliers_percentile = self.df.filter(perc_mask).height
        
        # Basic statistics
        min_val = col_data.min()
        max_val = col_data.max()
        
        return OutlierSummary(
            column=column,
            total_values=total_values,
            outliers_iqr=outliers_iqr,
            outliers_zscore=outliers_zscore,
            outliers_modified_zscore=outliers_modified_zscore,
            outliers_percentile=outliers_percentile,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_bound_iqr=lower_iqr,
            upper_bound_iqr=upper_iqr,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            p01=p01,
            p99=p99
        )
    
    def analyse_all_outliers(self) -> Dict[str, OutlierSummary]:
        """
        Analyse outliers for all numerical columns in the dataset.
        
        Returns:
            Dictionary mapping column names to OutlierSummary objects
        """
        if self.df is None:
            self.load_data()
        
        results = {}
        for column in self.numerical_columns:
            results[column] = self.analyse_column_outliers(column)
        
        self.outlier_results = results
        return results
    
    def get_outlier_recommendations(self, summary: OutlierSummary) -> List[str]:
        """
        Generate data cleaning recommendations based on outlier analysis.
        
        Args:
            summary: OutlierSummary for a column
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        total = summary.total_values
        
        # Calculate percentages
        iqr_pct = (summary.outliers_iqr / total) * 100
        zscore_pct = (summary.outliers_zscore / total) * 100
        mod_zscore_pct = (summary.outliers_modified_zscore / total) * 100
        perc_pct = (summary.outliers_percentile / total) * 100
        
        # Range analysis
        range_ratio = summary.max_val / summary.min_val if summary.min_val > 0 else float('inf')
        
        if iqr_pct > 10:
            recommendations.append(f"**High outlier percentage ({iqr_pct:.1f}%)** - Review data collection process")
        elif iqr_pct > 5:
            recommendations.append(f"**Moderate outliers ({iqr_pct:.1f}%)** - Consider capping or transformation")
        elif iqr_pct > 1:
            recommendations.append(f"**Some outliers ({iqr_pct:.1f}%)** - Investigate extreme values")
        
        if range_ratio > 1000:
            recommendations.append("**Extremely wide range** - Likely contains data entry errors or different units")
        elif range_ratio > 100:
            recommendations.append("**Wide value range** - Consider log transformation or outlier removal")
        
        # Consistency check between methods
        methods_agreement = len(set([
            summary.outliers_iqr, summary.outliers_zscore, 
            summary.outliers_modified_zscore, summary.outliers_percentile
        ]))
        
        if methods_agreement > 3:
            recommendations.append("**Methods show different outlier counts** - Data may have complex distribution")
        
        # Specific recommendations
        if summary.std > summary.mean * 2:
            recommendations.append("**High variability** - Consider data standardisation or robust scaling")
        
        if summary.min_val < 0 and summary.column.lower() in ['price', 'amount', 'cost', 'value']:
            recommendations.append("**Negative values detected** - Check for data entry errors in price/amount field")
        
        if len(recommendations) == 0:
            recommendations.append("**Clean data** - No significant outlier issues detected")
        
        return recommendations
    
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate Markdown report for outlier analysis.
        
        Args:
            output_path: Optional custom output directory path
            
        Returns:
            Path to the generated report file
        """
        if not self.outlier_results:
            self.analyse_all_outliers()
        
        # Set up output path
        if output_path is None:
            # Find project root (look for CLAUDE.md or hnm_data_analysis directory)
            current_path = Path.cwd()
            project_root = current_path
            
            # Look for project indicators
            while project_root != project_root.parent:
                if (project_root / "CLAUDE.md").exists() or (project_root / "hnm_data_analysis").exists():
                    break
                project_root = project_root.parent
            
            output_dir = project_root / "results" / "data_documentation" / "outliers"
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.file_path.stem}_outlier_analysis.md"
        report_path = output_dir / filename
        
        # Generate report content
        report_lines = [
            "# Outlier Analysis Report",
            "",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**File:** {self.file_path.name}",
            f"**Dataset Shape:** {self.df.height:,} rows × {self.df.width} columns",
            f"**Numerical Columns Analysed:** {len(self.numerical_columns)}",
            "",
            "## 📊 Executive Summary",
            "",
            self._generate_executive_summary(),
            "",
            "## 🔍 Outlier Detection Methods",
            "",
            "This analysis employs four complementary statistical methods:",
            "",
            "1. **Interquartile Range (IQR)**: Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR",
            "2. **Z-Score**: Values with |z-score| > 3.0 standard deviations from mean", 
            "3. **Modified Z-Score**: Uses median absolute deviation, |modified z-score| > 3.5",
            "4. **Percentile**: Values below 1st percentile or above 99th percentile",
            "",
            "## 📈 Detailed Analysis by Column",
            ""
        ]
        
        # Add detailed analysis for each column
        for column, summary in self.outlier_results.items():
            report_lines.extend(self._generate_column_section(column, summary))
        
        # Add methodology and recommendations
        report_lines.extend([
            "## 💡 Data Cleaning Recommendations",
            "",
            "### Priority Actions",
            ""
        ])
        
        # Collect all recommendations
        all_recommendations = []
        for column, summary in self.outlier_results.items():
            column_recs = self.get_outlier_recommendations(summary)
            for rec in column_recs:
                if rec not in all_recommendations and "Clean data" not in rec:
                    all_recommendations.append(f"- **{column}**: {rec}")
        
        if all_recommendations:
            report_lines.extend(all_recommendations)
        else:
            report_lines.append("- **Overall**: Dataset appears clean with minimal outlier concerns")
        
        report_lines.extend([
            "",
            "### Methodology Notes",
            "",
            "- **IQR Method**: Most robust for skewed distributions",
            "- **Z-Score**: Assumes normal distribution, sensitive to extreme outliers",
            "- **Modified Z-Score**: More robust than standard z-score, uses median",
            "- **Percentile**: Simple and intuitive, good for business rules",
            "",
            "### Next Steps",
            "",
            "1. **Investigate** extreme values identified by multiple methods",
            "2. **Validate** business logic for outlier values",  
            "3. **Consider** transformation techniques (log, Box-Cox) for skewed data",
            "4. **Document** outlier handling decisions for reproducibility",
            "",
            "---",
            "",
            "_Report generated using H&M Data Analytics - Outlier Analysis Module_"
        ])
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Return relative path from project root
        try:
            relative_path = report_path.relative_to(project_root)
            return str(relative_path)
        except (ValueError, NameError):
            # Fallback if project_root not found
            return str(report_path.relative_to(Path.cwd()))
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        total_columns = len(self.outlier_results)
        if total_columns == 0:
            return "No numerical columns found for analysis."
        
        # Calculate summary statistics
        high_outlier_cols = []
        moderate_outlier_cols = []
        clean_cols = []
        
        for column, summary in self.outlier_results.items():
            iqr_pct = (summary.outliers_iqr / summary.total_values) * 100
            if iqr_pct > 5:
                high_outlier_cols.append(f"{column} ({iqr_pct:.1f}%)")
            elif iqr_pct > 1:
                moderate_outlier_cols.append(f"{column} ({iqr_pct:.1f}%)")
            else:
                clean_cols.append(column)
        
        lines = []
        
        if high_outlier_cols:
            lines.append(f"**🚨 High Priority**: {len(high_outlier_cols)} columns with >5% outliers:")
            lines.extend([f"- {col}" for col in high_outlier_cols[:5]])  # Show first 5
            if len(high_outlier_cols) > 5:
                lines.append(f"- ... and {len(high_outlier_cols) - 5} more")
            lines.append("")
        
        if moderate_outlier_cols:
            lines.append(f"**⚠️ Moderate Priority**: {len(moderate_outlier_cols)} columns with 1-5% outliers:")
            lines.extend([f"- {col}" for col in moderate_outlier_cols[:3]])  # Show first 3
            if len(moderate_outlier_cols) > 3:
                lines.append(f"- ... and {len(moderate_outlier_cols) - 3} more")
            lines.append("")
        
        if clean_cols:
            lines.append(f"**✅ Clean Data**: {len(clean_cols)} columns with <1% outliers")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _generate_column_section(self, column: str, summary: OutlierSummary) -> List[str]:
        """Generate detailed section for a single column."""
        lines = [
            f"### {column}",
            "",
            "#### Statistical Summary",
            "",
            "| Metric | Value |",
            "| ------ | ----- |",
            f"| Total Values | {summary.total_values:,} |",
            f"| Min Value | {summary.min_val:.6f} |",
            f"| Max Value | {summary.max_val:.6f} |",
            f"| Mean | {summary.mean:.6f} |",
            f"| Standard Deviation | {summary.std:.6f} |",
            f"| Q1 (25th percentile) | {summary.q1:.6f} |",
            f"| Q3 (75th percentile) | {summary.q3:.6f} |",
            f"| IQR | {summary.iqr:.6f} |",
            "",
            "#### Outlier Detection Results",
            "",
            "| Method | Outliers | Percentage | Bounds/Threshold |",
            "| ------ | -------- | ---------- | ---------------- |",
            f"| IQR (1.5×) | {summary.outliers_iqr:,} | {(summary.outliers_iqr/summary.total_values)*100:.2f}% | [{summary.lower_bound_iqr:.6f}, {summary.upper_bound_iqr:.6f}] |",
            f"| Z-Score (±3.0) | {summary.outliers_zscore:,} | {(summary.outliers_zscore/summary.total_values)*100:.2f}% | μ±3σ |",
            f"| Modified Z-Score (±3.5) | {summary.outliers_modified_zscore:,} | {(summary.outliers_modified_zscore/summary.total_values)*100:.2f}% | MAD-based |",
            f"| Percentile (1%-99%) | {summary.outliers_percentile:,} | {(summary.outliers_percentile/summary.total_values)*100:.2f}% | [{summary.p01:.6f}, {summary.p99:.6f}] |",
            "",
            "#### Recommendations",
            ""
        ]
        
        recommendations = self.get_outlier_recommendations(summary)
        lines.extend([f"- {rec}" for rec in recommendations])
        lines.extend(["", "---", ""])
        
        return lines


def generate_outlier_report(file_path: str, sample_size: Optional[int] = None, 
                           output_path: Optional[str] = None) -> str:
    """
    Convenience function to generate outlier analysis report.
    
    Args:
        file_path: Path to the CSV or Parquet file
        sample_size: Optional sample size for large datasets
        output_path: Optional custom output directory path
        
    Returns:
        Path to the generated report file
        
    Example:
        ```python
        from hnm_data_analysis.data_understanding.outlier_analysis import generate_outlier_report
        
        # Generate outlier report (saves to results/data_documentation/outliers/)
        report_path = generate_outlier_report("data/processed/transactions_last_3_months.parquet")
        
        # With sampling for large datasets
        report_path = generate_outlier_report("data/raw/transactions_train.csv", sample_size=100000)
        
        # Custom output path
        report_path = generate_outlier_report("data/processed/customers.parquet", 
                                             output_path="reports/outliers/")
        ```
    """
    analyser = OutlierAnalyser(file_path, sample_size)
    return analyser.generate_markdown_report(output_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python outlier_analysis.py <file_path> [sample_size] [output_path]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        report_path = generate_outlier_report(file_path, sample_size, output_path)
        print(f"Outlier analysis report generated: {report_path}")
    except Exception as e:
        print(f"Error generating outlier report: {e}")
        sys.exit(1)