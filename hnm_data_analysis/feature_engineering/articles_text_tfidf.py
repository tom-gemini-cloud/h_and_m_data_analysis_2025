"""
Article description TF-IDF vectorisation (with optional lemmatisation and stemming)

This module loads the cleaned articles dataset, preprocesses the `detail_desc`
text field, vectorises it with TF-IDF, and optionally produces SVD (LSA)
embeddings. Outputs are saved under a specified directory for later use in
clustering.

Design choices:
- Polars for data I/O
- spaCy for lemmatisation
- NLTK Snowball stemmer for optional stemming
- scikit-learn for TF-IDF and TruncatedSVD

Outputs (by default):
- tfidf_features.npz            (sparse TF-IDF matrix)
- vectorizer.joblib             (fitted TfidfVectorizer)
- article_id_index.csv          (row index mapping)
- config.json                   (preprocessing + TF-IDF configuration)

Optional when --include-svd is specified:
- svd_embeddings.parquet        (dense SVD/LSA embeddings with columns svd_001..svd_k)
- svd_model.joblib              (fitted TruncatedSVD)

The module attempts to auto-detect a reasonable input file if none is provided.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import polars as pl
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


# Optional heavy imports guarded at runtime
def _ensure_spacy_model(language: str) -> None:
    """
    Ensure a spaCy model for the given language is available. Tries to load,
    otherwise attempts to download a small core model.

    Currently supports English ("en"). For other languages, users should install
    the relevant model manually.
    """
    import importlib

    lang = language.lower()
    if lang.startswith("en"):
        model_name = "en_core_web_sm"
    else:
        # Default guess; users can install a proper model manually
        model_name = f"{lang}_core_web_sm"

    try:
        importlib.import_module(model_name)
    except Exception:
        try:
            from spacy.cli import download as spacy_download

            print(f"spaCy model '{model_name}' not found. Downloading...")
            spacy_download(model_name)
        except Exception as e:  # pragma: no cover - network dependent branch
            warnings.warn(
                f"Could not download spaCy model '{model_name}'. "
                f"Install manually (python -m spacy download {model_name}). Error: {e}"
            )


def _load_spacy(language: str):
    import spacy

    _ensure_spacy_model(language)
    lang = language.lower()
    if lang.startswith("en"):
        model_name = "en_core_web_sm"
    else:
        model_name = f"{lang}_core_web_sm"

    # Disable heavy components for speed; keep tagger/lemmatizer
    try:
        nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])
    except Exception:
        # Fallback to a lightweight blank pipeline. Only add a lemmatiser if
        # the required lookup tables are available to avoid runtime errors.
        nlp = spacy.blank(lang if len(lang) == 2 else "en")
        try:
            # spacy-lookups-data provides the tables needed for rule-based lemmatisation
            import spacy_lookups_data  # type: ignore

            if "lemmatizer" not in nlp.pipe_names:
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
            # Initialise to load lookups
            try:
                nlp.initialize()
            except Exception:
                # If initialisation fails, remove the lemmatiser to keep pipeline usable
                if "lemmatizer" in nlp.pipe_names:
                    nlp.remove_pipe("lemmatizer")
        except Exception:
            # No lookups available; skip adding lemmatiser to prevent [E1004]
            pass
    return nlp


def _get_stopwords(language: str, extra: Optional[Iterable[str]] = None) -> set[str]:
    """Return a set of stopwords for a language, combining spaCy + custom."""
    try:
        import spacy

        nlp = spacy.blank(language if len(language) == 2 else language[:2])
        stopwords = set(nlp.Defaults.stop_words)
    except Exception:
        stopwords = set()
    # Always exclude the sentinel
    stopwords.update({"no_description"})
    if extra:
        stopwords.update({w.lower() for w in extra})
    return stopwords


def _normalise_whitespace(text: str) -> str:
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
class TfidfConfig:
    max_features: int
    min_df: int | float
    max_df: float
    ngram_range: Tuple[int, int]
    lowercase: bool = False
    token_pattern: str = r"(?u)\b\w\w+\b"


class ArticleDescriptionVectoriser:
    def __init__(
        self,
        input_path: Optional[str] = None,
        text_column: str = "detail_desc",
        id_column: str = "article_id",
        language: str = "en",
        use_lemmatise: bool = True,
        use_stem: bool = False,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.input_path = input_path
        self.text_column = text_column
        self.id_column = id_column
        self.language = language
        self.use_lemmatise = use_lemmatise
        self.use_stem = use_stem
        self.extra_stopwords = set(extra_stopwords) if extra_stopwords else set()

        self.df: Optional[pl.DataFrame] = None
        self.article_ids: List[Any] = []
        self.cleaned_texts: List[str] = []

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[csr_matrix] = None

        self.svd_model: Optional[TruncatedSVD] = None
        self.svd_embeddings: Optional[Any] = None  # numpy array

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
    def _build_texts(self, batch_size: int = 1000, n_process: int = 1) -> List[str]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data().")

        texts = self.df.get_column(self.text_column).to_list()
        self.article_ids = self.df.get_column(self.id_column).to_list()

        # Lowercase and quick clean before NLP
        texts = [
            _normalise_whitespace(str(t).lower())
            for t in texts
        ]

        if not self.use_lemmatise and not self.use_stem:
            # Light cleaning only (remove non-letters, collapse whitespace)
            cleaned = [re.sub(r"[^a-z\s]", " ", t) for t in texts]
            cleaned = [
                _normalise_whitespace(re.sub(r"\b\w\b", " ", c))  # drop single chars
                for c in cleaned
            ]
            return cleaned

        # spaCy lemmatisation pipeline
        nlp = _load_spacy(self.language)
        stopwords = _get_stopwords(self.language, self.extra_stopwords)

        # Optional stemmer
        stemmer = None
        if self.use_stem:
            try:
                from nltk.stem.snowball import SnowballStemmer

                stem_lang = "english" if self.language.lower().startswith("en") else self.language
                stemmer = SnowballStemmer(stem_lang)
            except Exception:
                warnings.warn(
                    "NLTK SnowballStemmer not available for language; proceeding without stemming."
                )

        cleaned_docs: List[str] = []
        use_lemma = self.use_lemmatise and ("lemmatizer" in nlp.pipe_names)
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            tokens: List[str] = []
            for token in doc:
                # Keep alphabetic tokens only, skip stopwords, very short tokens
                if not token.is_alpha:
                    continue
                lemma = token.lemma_.lower() if use_lemma else token.text.lower()
                if len(lemma) < 2:
                    continue
                if lemma in stopwords:
                    continue
                if stemmer is not None:
                    try:
                        lemma = stemmer.stem(lemma)
                    except Exception:
                        pass
                tokens.append(lemma)

            cleaned_docs.append(" ".join(tokens))

        return cleaned_docs

    def clean_text_series(self, batch_size: int = 1000, n_process: int = 1) -> List[str]:
        cleaned = self._build_texts(batch_size=batch_size, n_process=n_process)
        # Final pass to normalise whitespace
        cleaned = [_normalise_whitespace(c) for c in cleaned]
        self.cleaned_texts = cleaned
        print(f"Prepared cleaned texts: {len(self.cleaned_texts):,}")
        return cleaned

    # --------------------- TF-IDF vectorisation ---------------------
    @staticmethod
    def _suggest_tfidf_defaults(num_docs: int) -> TfidfConfig:
        min_df = max(5, round(0.001 * max(num_docs, 1)))
        # Cap lower bound of min_df to 1 for very small corpora
        if num_docs < 5000:
            min_df = max(1, round(0.002 * max(num_docs, 1)))
        cfg = TfidfConfig(
            max_features=50000 if num_docs >= 100000 else 20000,
            min_df=min_df,
            max_df=0.8,
            ngram_range=(1, 2),
            lowercase=False,
        )
        return cfg

    def fit_transform(self, **tfidf_kwargs: Any) -> csr_matrix:
        if not self.cleaned_texts:
            raise ValueError("Cleaned texts are empty. Call clean_text_series() first.")

        defaults = self._suggest_tfidf_defaults(len(self.cleaned_texts))
        # Merge user kwargs over defaults
        params = asdict(defaults)
        params.update(tfidf_kwargs)

        self.vectorizer = TfidfVectorizer(**params)
        print(
            f"Fitting TF-IDF: max_features={params['max_features']}, min_df={params['min_df']}, "
            f"max_df={params['max_df']}, ngram_range={params['ngram_range']}"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_texts)
        print(
            f"TF-IDF shape: {self.tfidf_matrix.shape[0]:,} docs x {self.tfidf_matrix.shape[1]:,} terms"
        )
        return self.tfidf_matrix

    def transform(self, new_texts: Sequence[str]) -> csr_matrix:
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        return self.vectorizer.transform(new_texts)

    # --------------------- SVD / LSA embeddings ---------------------
    def fit_svd(
        self,
        n_components: int = 200,
        random_state: int = 42,
        normalize: bool = True,
    ) -> Any:
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not available. Call fit_transform() first.")

        self.svd_model = TruncatedSVD(n_components=n_components, random_state=random_state)
        print(f"Fitting TruncatedSVD with n_components={n_components} ...")
        emb = self.svd_model.fit_transform(self.tfidf_matrix)
        if normalize:
            emb = Normalizer(copy=False).fit_transform(emb)
        self.svd_embeddings = emb
        print(f"SVD embeddings shape: {emb.shape[0]:,} x {emb.shape[1]:,}")
        return emb

    # --------------------------- Save outputs ---------------------------
    def _ensure_outdir(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

    def save_features(self, output_dir: str) -> None:
        if self.tfidf_matrix is None or self.vectorizer is None:
            raise ValueError("Nothing to save. Run fit_transform() first.")
        self._ensure_outdir(output_dir)

        tfidf_path = os.path.join(output_dir, "tfidf_features.npz")
        vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
        index_path = os.path.join(output_dir, "article_id_index.csv")
        config_path = os.path.join(output_dir, "config.json")

        print(f"Saving TF-IDF matrix to: {tfidf_path}")
        save_npz(tfidf_path, self.tfidf_matrix)
        print(f"Saving vectorizer to: {vectorizer_path}")
        joblib.dump(self.vectorizer, vectorizer_path)

        print(f"Saving article_id index to: {index_path}")
        pl.DataFrame({self.id_column: self.article_ids}).write_csv(index_path)

        cfg = {
            "input_path": self.input_path,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "language": self.language,
            "use_lemmatise": self.use_lemmatise,
            "use_stem": self.use_stem,
            "tfidf_params": _json_safe(self.vectorizer.get_params()),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_svd_embeddings(self, output_dir: str, filename: str = "svd_embeddings.parquet") -> None:
        if self.svd_embeddings is None or self.svd_model is None:
            raise ValueError("SVD embeddings not available. Call fit_svd() first.")
        self._ensure_outdir(output_dir)

        svd_path = os.path.join(output_dir, filename)
        model_path = os.path.join(output_dir, "svd_model.joblib")

        cols = [f"svd_{i:03d}" for i in range(1, self.svd_embeddings.shape[1] + 1)]
        df = pl.DataFrame({self.id_column: self.article_ids})
        for i, col in enumerate(cols):
            df = df.with_columns(pl.Series(name=col, values=self.svd_embeddings[:, i]))

        print(f"Saving SVD embeddings to: {svd_path}")
        df.write_parquet(svd_path) if svd_path.endswith(".parquet") else df.write_csv(svd_path)

        print(f"Saving SVD model to: {model_path}")
        joblib.dump(self.svd_model, model_path)

    # --------------------------- High-level API ---------------------------
    def process(
        self,
        output_dir: Optional[str] = None,
        include_svd: bool = False,
        svd_components: int = 200,
        svd_normalize: bool = True,
        nlp_batch_size: int = 1000,
        nlp_n_process: int = 1,
        **tfidf_kwargs: Any,
    ) -> Tuple[csr_matrix, Optional[Any]]:
        self.load_data()
        self.clean_text_series(batch_size=nlp_batch_size, n_process=nlp_n_process)
        self.fit_transform(**tfidf_kwargs)

        emb = None
        if include_svd:
            emb = self.fit_svd(n_components=svd_components, normalize=svd_normalize)

        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(base, "data", "features")

        self.save_features(output_dir)
        if include_svd and emb is not None:
            self.save_svd_embeddings(output_dir)

        return self.tfidf_matrix, emb


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Preprocess article descriptions and build TF-IDF features (optionally SVD/LSA embeddings)."
        )
    )
    p.add_argument("--input-path", type=str, default=None, help="Path to input articles file (csv/parquet)")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs")
    p.add_argument("--language", type=str, default="en", help="Language code for lemmatisation/stopwords (e.g., 'en')")
    p.add_argument("--no-lemmatiser", action="store_true", help="Disable lemmatisation")
    p.add_argument("--use-stem", action="store_true", help="Enable Snowball stemming (after lemmatisation if enabled)")
    p.add_argument("--include-svd", action="store_true", help="Also compute and save SVD embeddings")
    p.add_argument("--svd-components", type=int, default=200, help="Number of SVD components")
    p.add_argument("--no-svd-normalize", action="store_true", help="Disable L2 normalization after SVD")

    # TF-IDF overrides
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--min-df", type=float, default=None)
    p.add_argument("--max-df", type=float, default=None)
    p.add_argument("--ngram-min", type=int, default=None)
    p.add_argument("--ngram-max", type=int, default=None)

    p.add_argument("--nlp-batch-size", type=int, default=1000)
    p.add_argument("--nlp-n-process", type=int, default=1)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    vectoriser = ArticleDescriptionVectoriser(
        input_path=args.input_path,
        language=args.language,
        use_lemmatise=not args.no_lemmatiser,
        use_stem=args.use_stem,
    )

    # Build TF-IDF kwargs from CLI overrides
    tfidf_kwargs: Dict[str, Any] = {}
    if args.max_features is not None:
        tfidf_kwargs["max_features"] = args.max_features
    if args.min_df is not None:
        tfidf_kwargs["min_df"] = args.min_df
    if args.max_df is not None:
        tfidf_kwargs["max_df"] = args.max_df
    if args.ngram_min is not None or args.ngram_max is not None:
        nmin = args.ngram_min if args.ngram_min is not None else 1
        nmax = args.ngram_max if args.ngram_max is not None else max(1, nmin)
        tfidf_kwargs["ngram_range"] = (nmin, nmax)

    vectoriser.process(
        output_dir=args.output_dir,
        include_svd=args.include_svd,
        svd_components=args.svd_components,
        svd_normalize=not args.no_svd_normalize,
        nlp_batch_size=args.nlp_batch_size,
        nlp_n_process=args.nlp_n_process,
        **tfidf_kwargs,
    )


if __name__ == "__main__": 
    main()

 