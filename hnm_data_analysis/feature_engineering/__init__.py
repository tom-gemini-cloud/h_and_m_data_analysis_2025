"""
Feature engineering module for H&M data analysis.

This package exposes feature construction utilities, including text-based
representations for articles, categorical feature encoding, and combined
feature generation for downstream modelling and clustering.
"""

from .customer_features import *  # re-export existing public features if any

# Vectorisation utilities
from .articles_text_tfidf import ArticleDescriptionVectoriser
from .articles_text_bert import ArticleDescriptionBertEmbedder

# Combined features utilities
from .combined_features_svd import CombinedFeaturesEngine
from .combined_features_bert import CombinedBertArticleFeatures

__all__ = [
    # keep wildcards minimal; explicitly list known public API
    "ArticleDescriptionVectoriser",
    "ArticleDescriptionBertEmbedder",
    "CombinedFeaturesEngine",
    "CombinedBertArticleFeatures",
]

