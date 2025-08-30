# Product Description Features for Customer Segmentation

This document outlines how to leverage the `detail_desc` field from the H&M articles dataset for advanced customer behavioural segmentation using text analysis techniques.

## Overview

The H&M articles dataset contains detailed product descriptions in the `detail_desc` field, providing rich textual information about product attributes, materials, styles, and features. This unstructured text data can be transformed into quantifiable customer preference features for sophisticated behavioural segmentation.

## TF-IDF Analysis for Customer Segmentation

### What TF-IDF Provides

**Term Frequency-Inverse Document Frequency (TF-IDF)** analysis on product descriptions would generate:

#### Customer Feature Vectors
- Numerical representation of each customer's preference for specific product attributes
- Weighted scores showing which descriptive terms are most important to each customer's purchasing behaviour
- Multi-dimensional vectors where each dimension represents a unique term from product descriptions

#### Example Customer Feature Vector
```
Customer A: [0.8, 0.0, 0.3, 0.6, 0.0, 0.9, 0.4, ...]
Customer B: [0.1, 0.7, 0.0, 0.2, 0.8, 0.0, 0.3, ...]

Where positions represent:
Position 0: "cotton" (A=0.8, B=0.1)
Position 1: "formal" (A=0.0, B=0.7) 
Position 2: "stretch" (A=0.3, B=0.0)
Position 3: "casual" (A=0.6, B=0.2)
Position 4: "slim-fit" (A=0.0, B=0.8)
Position 5: "breathable" (A=0.9, B=0.0)
Position 6: "denim" (A=0.4, B=0.3)
```

### Customer Insights from TF-IDF

#### Style Preference Clusters
- Customers grouped by similar vocabulary preferences
- **Classic vs Trendy**: Customers preferring "timeless", "classic" vs "on-trend", "contemporary"
- **Casual vs Formal**: Preference for "relaxed", "everyday" vs "smart", "professional"
- **Active vs Lifestyle**: "sporty", "performance" vs "comfortable", "leisure"

#### Material and Fabric Preferences
- **Natural fibres**: Cotton, linen, wool, silk preferences
- **Synthetic materials**: Polyester, elastane, nylon affinities
- **Fabric properties**: "breathable", "stretch", "moisture-wicking", "wrinkle-resistant"

#### Fit and Style Descriptors
- **Fit preferences**: "slim", "relaxed", "oversized", "fitted", "tailored"
- **Style characteristics**: "vintage", "minimalist", "bohemian", "contemporary"
- **Functional features**: "pockets", "lined", "reversible", "adjustable"

#### Quality and Premium Indicators
- **Premium descriptors**: "luxury", "premium", "crafted", "detailed"
- **Basic indicators**: "essential", "basic", "everyday", "simple"
- **Quality signals**: "durable", "high-quality", "reinforced"

## Customer Similarity Analysis

### Description Similarity Approaches

#### 1. Cosine Similarity on Aggregated TF-IDF
```python
# Process:
# 1. Concatenate all product descriptions for each customer
# 2. Calculate TF-IDF vectors for each customer's aggregate preferences
# 3. Measure cosine similarity between customer vectors
similarity_score = cosine_similarity(customer_A_tfidf, customer_B_tfidf)
```

**Advantages:**
- Captures overall style vocabulary preferences
- Computationally efficient for large customer base
- Interpretable similarity scores (0-1 range)

#### 2. Average Description Embeddings
```python
# Process:
# 1. Generate embeddings for each product description
# 2. Average embeddings of all products each customer purchased
# 3. Calculate cosine/euclidean distance between customer profiles
```

**Advantages:**
- Captures semantic meaning beyond exact word matches
- Better handling of synonyms and related terms
- More sophisticated representation of style preferences

#### 3. Jaccard Similarity on Keywords
```python
# Process:
# 1. Extract key terms from each customer's product descriptions
# 2. Compare overlap of preferred terms between customers
jaccard_similarity = |terms_A ∩ terms_B| / |terms_A ∪ terms_B|
```

**Advantages:**
- Simple to compute and interpret
- Good for identifying customers with overlapping specific preferences
- Useful for binary presence/absence of key terms

#### 4. Semantic Similarity with Sentence Transformers
```python
# Process:
# 1. Use BERT-based models on concatenated descriptions
# 2. Generate high-dimensional semantic embeddings
# 3. Calculate similarity in semantic space
```

**Advantages:**
- Understands context and synonyms
- Captures nuanced style relationships
- State-of-the-art text similarity performance

## Clustering Approaches for Description Features

### K-Means Clustering

#### Suitability for TF-IDF Vectors
**Advantages:**
- Works well with continuous numerical TF-IDF features
- Efficient for large customer datasets (1.37M customers)
- Cluster centres interpretable as "average style profiles"
- Scalable and well-established methodology

**Limitations:**
- Assumes spherical clusters (customers may have complex preference patterns)
- Sensitive to high dimensionality (TF-IDF can create thousands of features)
- Requires predefined number of clusters (k)

#### Implementation Recommendations
1. **Dimensionality reduction first**: Apply PCA or t-SNE before clustering
2. **Feature selection**: Keep only most discriminative TF-IDF terms
3. **Optimal k selection**: Use elbow method and silhouette analysis
4. **Text preprocessing**: Remove common words, standardise terminology

### Alternative Clustering Methods

#### Hierarchical Clustering
- Reveals natural customer preference hierarchies
- No need to predefine number of clusters
- Provides dendrogram for exploring cluster relationships
- Better for understanding preference structure

#### DBSCAN (Density-Based Clustering)
- Finds clusters of varying shapes and densities
- Handles outliers and noise effectively
- Automatically determines number of clusters
- Good for identifying niche customer segments

#### Gaussian Mixture Models
- More flexible cluster shapes than k-means
- Provides probability of cluster membership
- Better handles overlapping customer segments
- Captures uncertainty in cluster assignments

## Segmentation Applications

### Lifestyle Segments
- **Active/Performance**: Customers preferring "sporty", "performance", "breathable"
- **Professional/Formal**: Preference for "smart", "tailored", "professional"
- **Casual/Comfort**: Affinity for "relaxed", "comfortable", "everyday"
- **Fashion-Forward**: Interest in "trendy", "contemporary", "statement"

### Quality Consciousness Segments
- **Premium Seekers**: High affinity for luxury and quality descriptors
- **Value Conscious**: Preference for basic and essential items
- **Feature Focused**: Emphasis on functional attributes and performance

### Material and Fabric Segments
- **Natural Fibre Enthusiasts**: Strong preference for cotton, linen, wool
- **Performance Material Users**: Attraction to technical and synthetic fabrics
- **Comfort Prioritisers**: Focus on soft, breathable, and comfortable materials

### Style Consistency Analysis
- **Consistent Stylists**: Customers with coherent style vocabulary across purchases
- **Experimental Shoppers**: Diverse style preferences and broad vocabulary
- **Niche Specialists**: Strong focus on specific style categories or attributes

## Implementation Strategy

### Phase 1: Basic TF-IDF Analysis
1. **Data preparation**: Clean and preprocess `detail_desc` text
2. **TF-IDF calculation**: Generate customer feature vectors
3. **Dimensionality reduction**: Apply PCA to manage feature space
4. **Initial clustering**: Use k-means with multiple k values

### Phase 2: Advanced Similarity Analysis
1. **Cosine similarity**: Calculate pairwise customer similarities
2. **Similarity-based clustering**: Use similarity matrices for clustering
3. **Customer profiling**: Identify top terms for each cluster
4. **Validation**: Compare with other behavioural segments

### Phase 3: Semantic Enhancement
1. **Embedding models**: Implement sentence transformers
2. **Semantic clustering**: Compare with TF-IDF results
3. **Hybrid approaches**: Combine multiple similarity measures
4. **Recommendation systems**: Use similarity for product recommendations

## Technical Considerations

### Preprocessing Requirements
- **Text cleaning**: Remove punctuation, standardise formatting
- **Stop word removal**: Filter common, non-informative terms
- **Stemming/Lemmatisation**: Normalise word variations
- **Minimum frequency filtering**: Remove very rare terms

### Dimensionality Management
- **Feature selection**: Keep top N most informative terms
- **PCA/SVD**: Reduce dimensions while preserving variance
- **Sparse matrix handling**: Efficient storage for large TF-IDF matrices

### Scalability Considerations
- **Batch processing**: Handle large customer dataset efficiently
- **Memory management**: Optimise for 1.37M customer vectors
- **Computational efficiency**: Balance accuracy with processing time

## Expected Outcomes

### Customer Insights
- **Style preference profiles** for each customer segment
- **Cross-segment analysis** comparing description-based vs traditional segments
- **Personalisation opportunities** based on textual preferences

### Business Applications
- **Targeted marketing**: Messaging aligned with style vocabularies
- **Product recommendations**: Based on description similarity
- **Inventory planning**: Understanding demand for specific attributes
- **New product development**: Identifying gaps in style descriptors

The `detail_desc` field provides a rich source of unstructured data that can significantly enhance customer segmentation beyond traditional demographic and transactional approaches, enabling more nuanced understanding of customer style preferences and shopping motivations.