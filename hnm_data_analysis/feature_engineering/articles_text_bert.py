"""
Article description BERT embeddings using sentence transformers

This module loads the cleaned articles dataset, preprocesses the `detail_desc`
text field, and generates semantic embeddings using BERT-based sentence transformers.
Outputs are saved under a specified directory for later use in clustering.

Design choices:
- Polars for data I/O
- sentence-transformers for BERT embeddings
- scikit-learn for optional dimensionality reduction
- Lightweight text preprocessing (no heavy NLP)

Outputs (by default):
- bert_embeddings.parquet        (dense BERT embeddings)
- bert_model_info.json           (model configuration and metadata)
- article_id_index.csv           (row index mapping)
- config.json                    (preprocessing configuration)

Optional when --include-pca is specified:
- pca_embeddings.parquet         (reduced PCA embeddings)
- pca_model.joblib               (fitted PCA model)

The module attempts to auto-detect a reasonable input file if none is provided.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


def _normalise_whitespace(text: str) -> str:
    """Normalise whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def _json_safe(value: Any) -> Any:
    """Convert objects to JSON-safe representations (recursively)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass
    # Dict
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    # List/Tuple/Set
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # Fallback to string representation
    return str(value)


@dataclass
class BertConfig:
    model_name: str
    max_length: int
    batch_size: int
    device: str
    normalize_embeddings: bool = True


class ArticleDescriptionBertEmbedder:
    def __init__(
        self,
        input_path: Optional[str] = None,
        text_column: str = "detail_desc",
        id_column: str = "article_id",
        model_name: str = "all-MiniLM-L6-v2",
        max_length: int = 128,
        batch_size: int = 32,
        device: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        self.input_path = input_path
        self.text_column = text_column
        self.id_column = id_column
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.df: Optional[pl.DataFrame] = None
        self.article_ids: List[Any] = []
        self.cleaned_texts: List[str] = []
        self.bert_embeddings: Optional[np.ndarray] = None
        self.model = None

        self.pca_model: Optional[PCA] = None
        self.pca_embeddings: Optional[np.ndarray] = None

    # ----------------------- Data I/O -----------------------
    def _auto_detect_input(self) -> Optional[str]:
        """Try to find a reasonable default input if none provided."""
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates = [
            os.path.join(base, "data", "processed", "articles_last_3_months_cleaned.parquet"),
            os.path.join(base, "data", "processed", "articles_last_3_months_cleaned.csv"),
            os.path.join(base, "data", "cleaned", "articles_last_3_months_cleaned.parquet"),
            os.path.join(base, "data", "cleaned", "articles_last_3_months_cleaned.csv"),
            os.path.join(base, "data", "processed", "articles_last_3_months.parquet"),
            os.path.join(base, "data", "processed", "articles_last_3_months.csv"),
            os.path.join(base, "data", "raw", "articles.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def load_data(self) -> pl.DataFrame:
        """Load articles data using Polars and filter valid descriptions."""
        path = self.input_path or self._auto_detect_input()
        if path is None:
            raise FileNotFoundError(
                "No input path provided and no default articles file found in data/."
            )
        self.input_path = path

        print(f"Loading articles from: {path}")
        if path.endswith(".parquet"):
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path)

        if self.text_column not in df.columns or self.id_column not in df.columns:
            raise KeyError(
                f"Input data must contain columns '{self.id_column}' and '{self.text_column}'."
            )

        # Standardise NO_DESCRIPTION handling and remove nulls
        df = df.with_columns(
            pl.col(self.text_column)
            .cast(pl.Utf8)
            .fill_null("")
            .str.strip_chars()
        )

        # Filter out NO_DESCRIPTION rows (case-insensitive) and empties
        df = df.filter(
            (pl.col(self.text_column).str.to_lowercase() != "no_description")
            & (pl.col(self.text_column).str.len_chars() > 0)
        )

        self.df = df
        print(f"Articles with valid descriptions: {df.height:,}")
        return df

    # -------------------- Text preprocessing --------------------
    def _clean_texts(self) -> List[str]:
        """Clean and prepare texts for BERT embedding."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data().")

        texts = self.df.get_column(self.text_column).to_list()
        self.article_ids = self.df.get_column(self.id_column).to_list()

        # Light cleaning for BERT (preserve semantic meaning)
        cleaned = []
        for text in texts:
            # Normalise whitespace
            text = _normalise_whitespace(str(text))
            
            # Basic cleaning: remove excessive punctuation but keep semantic structure
            text = re.sub(r'[^\w\s\-.,!?]', ' ', text)  # Keep basic punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalise whitespace again
            text = text.strip()
            
            # Handle empty texts after cleaning
            if not text:
                text = "no description available"
            
            cleaned.append(text)

        self.cleaned_texts = cleaned
        print(f"Prepared cleaned texts: {len(self.cleaned_texts):,}")
        return cleaned

    def clean_text_series(self) -> List[str]:
        """Clean text series and return prepared texts."""
        return self._clean_texts()

    # --------------------- BERT Embeddings ---------------------
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        print(f"Loading BERT model: {self.model_name}")
        
        # Auto-detect device
        if self.device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = self.device

        self.model = SentenceTransformer(self.model_name, device=device)
        print(f"Model loaded on device: {device}")
        return self.model

    def fit_transform(self, show_progress_bar: bool = True) -> np.ndarray:
        """Generate BERT embeddings for all texts."""
        if not self.cleaned_texts:
            raise ValueError("Cleaned texts are empty. Call clean_text_series() first.")

        if self.model is None:
            self._load_model()

        print(f"Generating BERT embeddings for {len(self.cleaned_texts):,} texts...")
        print(f"Model: {self.model_name}, Max length: {self.max_length}, Batch size: {self.batch_size}")

        # Generate embeddings
        self.bert_embeddings = self.model.encode(
            self.cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            max_length=self.max_length,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )

        print(f"BERT embeddings shape: {self.bert_embeddings.shape[0]:,} docs x {self.bert_embeddings.shape[1]:,} dimensions")
        return self.bert_embeddings

    def transform(self, new_texts: Sequence[str], show_progress_bar: bool = True) -> np.ndarray:
        """Generate embeddings for new texts using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_transform() first.")
        
        # Clean new texts
        cleaned_new_texts = []
        for text in new_texts:
            text = _normalise_whitespace(str(text))
            text = re.sub(r'[^\w\s\-.,!?]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                text = "no description available"
            cleaned_new_texts.append(text)

        return self.model.encode(
            cleaned_new_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            max_length=self.max_length,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )

    # --------------------- PCA Dimensionality Reduction ---------------------
    def fit_pca(
        self,
        n_components: int = 200,
        random_state: int = 42,
        normalize: bool = True,
    ) -> np.ndarray:
        """Apply PCA dimensionality reduction to BERT embeddings."""
        if self.bert_embeddings is None:
            raise ValueError("BERT embeddings not available. Call fit_transform() first.")

        self.pca_model = PCA(n_components=n_components, random_state=random_state)
        print(f"Fitting PCA with n_components={n_components} ...")
        emb = self.pca_model.fit_transform(self.bert_embeddings)
        
        if normalize:
            emb = Normalizer(copy=False).fit_transform(emb)
        
        self.pca_embeddings = emb
        print(f"PCA embeddings shape: {emb.shape[0]:,} x {emb.shape[1]:,}")
        print(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.3f}")
        return emb

    # --------------------------- Save outputs ---------------------------
    def _ensure_outdir(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

    def save_embeddings(self, output_dir: str) -> None:
        """Save BERT embeddings and metadata."""
        if self.bert_embeddings is None or self.model is None:
            raise ValueError("Nothing to save. Run fit_transform() first.")
        self._ensure_outdir(output_dir)

        embeddings_path = os.path.join(output_dir, "bert_embeddings.parquet")
        model_info_path = os.path.join(output_dir, "bert_model_info.json")
        index_path = os.path.join(output_dir, "article_id_index.csv")
        config_path = os.path.join(output_dir, "config.json")

        # Save embeddings as parquet
        print(f"Saving BERT embeddings to: {embeddings_path}")
        cols = [f"bert_{i:03d}" for i in range(1, self.bert_embeddings.shape[1] + 1)]
        df = pl.DataFrame({self.id_column: self.article_ids})
        for i, col in enumerate(cols):
            df = df.with_columns(pl.Series(name=col, values=self.bert_embeddings[:, i]))
        df.write_parquet(embeddings_path)

        # Save model info
        print(f"Saving model info to: {model_info_path}")
        model_info = {
            "model_name": self.model_name,
            "embedding_dimension": self.bert_embeddings.shape[1],
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
        }
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)

        # Save article ID index
        print(f"Saving article_id index to: {index_path}")
        pl.DataFrame({self.id_column: self.article_ids}).write_csv(index_path)

        # Save configuration
        cfg = {
            "input_path": self.input_path,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_pca_embeddings(self, output_dir: str, filename: str = "pca_embeddings.parquet") -> None:
        """Save PCA embeddings and model."""
        if self.pca_embeddings is None or self.pca_model is None:
            raise ValueError("PCA embeddings not available. Call fit_pca() first.")
        self._ensure_outdir(output_dir)

        pca_path = os.path.join(output_dir, filename)
        model_path = os.path.join(output_dir, "pca_model.joblib")

        # Save PCA embeddings
        cols = [f"pca_{i:03d}" for i in range(1, self.pca_embeddings.shape[1] + 1)]
        df = pl.DataFrame({self.id_column: self.article_ids})
        for i, col in enumerate(cols):
            df = df.with_columns(pl.Series(name=col, values=self.pca_embeddings[:, i]))

        print(f"Saving PCA embeddings to: {pca_path}")
        df.write_parquet(pca_path) if pca_path.endswith(".parquet") else df.write_csv(pca_path)

        # Save PCA model
        print(f"Saving PCA model to: {model_path}")
        joblib.dump(self.pca_model, model_path)

    # --------------------------- High-level API ---------------------------
    def process(
        self,
        output_dir: Optional[str] = None,
        include_pca: bool = False,
        pca_components: int = 200,
        pca_normalize: bool = True,
        show_progress_bar: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Complete pipeline: load, clean, embed, and optionally reduce dimensions."""
        self.load_data()
        self.clean_text_series()
        self.fit_transform(show_progress_bar=show_progress_bar)

        pca_emb = None
        if include_pca:
            pca_emb = self.fit_pca(n_components=pca_components, normalize=pca_normalize)

        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(base, "data", "features")

        self.save_embeddings(output_dir)
        if include_pca and pca_emb is not None:
            self.save_pca_embeddings(output_dir)

        return self.bert_embeddings, pca_emb


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Generate BERT embeddings for article descriptions using sentence transformers."
        )
    )
    p.add_argument("--input-path", type=str, default=None, help="Path to input articles file (csv/parquet)")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs")
    p.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", 
                   help="Sentence transformer model name")
    p.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    p.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    p.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    p.add_argument("--include-pca", action="store_true", help="Also compute and save PCA embeddings")
    p.add_argument("--pca-components", type=int, default=200, help="Number of PCA components")
    p.add_argument("--no-pca-normalize", action="store_true", help="Disable L2 normalization after PCA")
    p.add_argument("--no-progress-bar", action="store_true", help="Disable progress bar")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    embedder = ArticleDescriptionBertEmbedder(
        input_path=args.input_path,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
    )

    embedder.process(
        output_dir=args.output_dir,
        include_pca=args.include_pca,
        pca_components=args.pca_components,
        pca_normalize=not args.no_pca_normalize,
        show_progress_bar=not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
