"""
Combined BERT Features Module

This module joins cleaned article metadata with precomputed BERT embeddings
and saves the combined dataset for downstream modelling and clustering.

Defaults:
- Cleaned articles: data/cleaned/articles_last_3_months_cleaned.parquet
- BERT embeddings: data/features/bert/bert_embeddings.parquet
- Output directory: data/features/combined

Output:
- Parquet file containing `article_id`, BERT embedding columns (bert_XXX),
  and selected article metadata columns.

Notes:
- The module performs an inner join on `article_id` so only articles that have
  both metadata and embeddings are kept.
"""

from __future__ import annotations

import os
from typing import List, Optional

import polars as pl


DEFAULT_CLEANED_ARTICLES = "data/cleaned/articles_last_3_months_cleaned.parquet"
DEFAULT_BERT_EMBEDDINGS = "data/features/bert/bert_embeddings.parquet"
DEFAULT_OUTPUT_DIR = "data/features/combined"
DEFAULT_OUTPUT_FILENAME = "articles_with_bert_embeddings.parquet"


class CombinedBertArticleFeatures:
    """Join cleaned article metadata with BERT embeddings and save the result."""

    def __init__(
        self,
        cleaned_articles_path: str = DEFAULT_CLEANED_ARTICLES,
        bert_embeddings_path: str = DEFAULT_BERT_EMBEDDINGS,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        include_metadata_columns: Optional[List[str]] = None,
    ) -> None:
        self.cleaned_articles_path = cleaned_articles_path
        self.bert_embeddings_path = bert_embeddings_path
        self.output_dir = output_dir
        self.include_metadata_columns = include_metadata_columns

        self.articles_df: Optional[pl.DataFrame] = None
        self.bert_df: Optional[pl.DataFrame] = None

    def _load_inputs(self) -> None:
        if not os.path.exists(self.cleaned_articles_path):
            raise FileNotFoundError(
                f"Cleaned articles not found: {self.cleaned_articles_path}"
            )
        if not os.path.exists(self.bert_embeddings_path):
            raise FileNotFoundError(
                f"BERT embeddings not found: {self.bert_embeddings_path}"
            )

        self.articles_df = pl.read_parquet(self.cleaned_articles_path)
        self.bert_df = pl.read_parquet(self.bert_embeddings_path)

        if "article_id" not in self.articles_df.columns:
            raise ValueError("Cleaned articles must contain 'article_id' column")
        if "article_id" not in self.bert_df.columns:
            raise ValueError("BERT embeddings must contain 'article_id' column")

        # Basic info
        print(f"Loaded cleaned articles: {self.articles_df.shape}")
        print(f"Loaded BERT embeddings: {self.bert_df.shape}")

    def _select_metadata_columns(self) -> List[str]:
        if self.articles_df is None:
            raise RuntimeError("Articles dataframe not loaded.")

        if self.include_metadata_columns is not None:
            selected = [c for c in self.include_metadata_columns if c in self.articles_df.columns]
        else:
            # A sensible default subset of metadata columns
            candidates = [
                "product_type_name",
                "product_group_name",
                "department_name",
                "section_name",
                "garment_group_name",
                "colour_group_name",
                "graphical_appearance_name",
            ]
            selected = [c for c in candidates if c in self.articles_df.columns]
        # Always keep article_id
        return ["article_id"] + selected

    def combine(self) -> pl.DataFrame:
        self._load_inputs()
        assert self.articles_df is not None and self.bert_df is not None

        meta_cols = self._select_metadata_columns()
        articles_subset = self.articles_df.select(meta_cols)

        # Identify BERT columns
        bert_cols = [c for c in self.bert_df.columns if c.startswith("bert_")]
        if not bert_cols:
            raise ValueError("No BERT embedding columns found (expected columns starting with 'bert_')")

        # Inner join to keep rows present in both
        combined = self.bert_df.join(articles_subset, on="article_id", how="inner")
        print(f"Combined dataset shape: {combined.shape}")
        print(f"Articles with embeddings and metadata: {combined.height:,}")
        return combined

    def save(self, combined: pl.DataFrame) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, DEFAULT_OUTPUT_FILENAME)
        combined.write_parquet(out_path)
        print(f"Saved combined dataset to: {out_path}")
        return out_path

    def run(self) -> str:
        combined = self.combine()
        return self.save(combined)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine cleaned article metadata with BERT embeddings"
    )
    parser.add_argument(
        "--cleaned-articles-path",
        type=str,
        default=DEFAULT_CLEANED_ARTICLES,
        help="Path to cleaned articles parquet",
    )
    parser.add_argument(
        "--bert-embeddings-path",
        type=str,
        default=DEFAULT_BERT_EMBEDDINGS,
        help="Path to BERT embeddings parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the combined dataset",
    )
    parser.add_argument(
        "--include-metadata-columns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of metadata columns to include in the output. "
            "Defaults to a sensible subset if not provided."
        ),
    )

    args = parser.parse_args(argv)

    job = CombinedBertArticleFeatures(
        cleaned_articles_path=args.cleaned_articles_path,
        bert_embeddings_path=args.bert_embeddings_path,
        output_dir=args.output_dir,
        include_metadata_columns=args.include_metadata_columns,
    )
    job.run()


if __name__ == "__main__":
    main()
