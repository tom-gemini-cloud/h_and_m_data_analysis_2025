"""
Utility script to generate a data understanding report for very large files.

This wrapper uses the existing `DataReportGenerator` but, for CSV inputs,
limits reading to the first N rows to keep memory usage manageable while
producing an identical report structure.

Usage:
    python -m hnm_data_analysis.data_understanding.generate_largefile_report \
        <file_path> [--output <output_path>] [--sample-size <n_rows>] [--full]

Notes:
    - For CSV files, only the first `sample_size` rows are loaded.
    - For Parquet, the file is read and then truncated to the first
      `sample_size` rows, which is typically efficient with columnar storage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_report_generator import generate_data_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a data understanding report for large datasets. "
            "For CSVs, reads only the first N rows so you can quickly "
            "visualise structure and quality without loading the entire file."
        )
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the CSV or Parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path for the Markdown report. "
            "Defaults to results/data_documentation/data_reports/<name>_data_report.md"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of rows to read from the top of the dataset. "
            "Omit to analyse the full file (may be slow and memory intensive)."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Analyse the entire file (equivalent to omitting --sample-size).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path: Path = args.file_path
    output_path: Path | None = args.output
    sample_size: int | None = None if args.full else args.sample_size

    report_path = generate_data_report(str(file_path), str(output_path) if output_path else None, sample_size)

    print("Data report generated successfully for large file!")
    print(f"Analysed file: {file_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()


