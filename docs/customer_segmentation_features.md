# Customer Segmentation Features for H&M Data Analysis

## Overview

This document outlines sensible customer segmentation features that can be generated from the H&M dataset to enable effective customer clustering and analysis. The features are organised by category and implementation priority to support retail analytics and marketing strategy development.

## 🛍️ RFM Analysis Features (Core Segmentation)

RFM analysis forms the foundation of customer segmentation by analysing customer behaviour across three key dimensions:

### Recency
- `days_since_last_purchase` - Number of days since most recent transaction
- `is_recent_customer` - Boolean flag indicating purchase in last 30 days

### Frequency
- `total_transactions` - Total number of purchases made by customer
- `avg_days_between_purchases` - Average time interval between transactions
- `purchase_frequency_score` - Standardised purchases per month metric

### Monetary
- `total_spent` - Total monetary value across all transactions
- `avg_order_value` - Average transaction amount per purchase
- `spending_tier` - Categorical grouping: Low/Medium/High spender

## 👕 Product Preference Features

Product-based features capture customer preferences and shopping behaviour patterns:

### Category Preferences
- `favourite_product_group` - Most frequently purchased product category
- `category_diversity` - Count of distinct categories purchased
- `seasonal_preference` - Bias towards Spring/Summer/Autumn/Winter collections

### Shopping Behaviour
- `avg_items_per_transaction` - Average basket size indicator
- `price_sensitivity` - Preference ratio for discounted versus full-price items
- `brand_loyalty` - Tendency to purchase from specific brands versus variety shopping

## 📊 Demographic & Behavioural Features

Customer profile features utilise demographic data and shopping patterns:

### Customer Profile
- `age_group` - Categorical age segmentation: Young/Adult/Senior (from customers.csv)
- `club_member_status` - H&M loyalty programme membership status
- `fashion_news_subscriber` - Newsletter subscription flag

### Shopping Patterns
- `preferred_channel` - Channel preference: Online versus in-store shopping
- `weekend_shopper` - Boolean flag for customers who primarily shop on weekends
- `impulse_vs_planned` - Shopping style: Single-item versus multi-item transactions

## 🔄 Lifecycle & Engagement Features

Features that capture customer relationship evolution and engagement levels:

### Customer Lifecycle
- `customer_tenure` - Number of months since first recorded purchase
- `lifecycle_stage` - Categorical stage: New/Active/At-Risk/Lost customer
- `purchase_trend` - Spending trajectory: Increasing/Stable/Decreasing

### Engagement Metrics
- `shopping_consistency` - Regularity and predictability of purchase behaviour
- `seasonal_activity` - Engagement across multiple seasonal periods
- `response_to_promotions` - Purchase behaviour during promotional periods

## 💡 Advanced Segmentation Features

Sophisticated features for enhanced customer insight and predictive analytics:

### Value-Based Features
- `customer_lifetime_value` - Projected CLV using historical purchase patterns
- `profit_margin_preference` - Tendency towards high versus low margin products
- `return_customer_score` - Probability score for customer retention

### Behavioural Clusters
- `exploration_score` - Propensity to try new categories and brands
- `fashion_forward_score` - Speed of adoption for latest trends and collections
- `bargain_hunter_score` - Focus on discounted and promotional items

## 🎯 Implementation Priority Framework

### Tier 1 (Essential Features)
**Priority:** High | **Complexity:** Low | **Business Impact:** High

- RFM analysis features (Recency, Frequency, Monetary)
- Basic demographic features (age groups, membership status)
- Core shopping metrics (total spent, transaction count)

**Rationale:** These features provide immediate segmentation value and are fundamental for retail customer analysis.

### Tier 2 (Enhancement Features)
**Priority:** Medium | **Complexity:** Medium | **Business Impact:** High

- Product preference features (category diversity, brand loyalty)
- Shopping pattern analysis (channel preferences, seasonal behaviour)
- Engagement metrics (consistency, promotional response)

**Rationale:** Enhance segmentation granularity and provide actionable insights for marketing campaigns.

### Tier 3 (Advanced Features)
**Priority:** Low | **Complexity:** High | **Business Impact:** Medium

- Lifecycle stage analysis and trend prediction
- Behavioural clustering and personality scoring
- Predictive customer lifetime value modelling

**Rationale:** Sophisticated features requiring advanced analytics but providing strategic insights for long-term planning.

## 📈 Segmentation Applications

These features enable implementation of industry-standard segmentation methodologies:

### Classic RFM Segmentation
- **Champions:** High RFM scores across all dimensions
- **Loyal Customers:** High frequency and monetary, moderate recency
- **At-Risk:** High monetary and frequency, low recency
- **New Customers:** High recency, low frequency and monetary

### Demographic Clustering
- **Life-stage Segments:** Age-based lifestyle groupings
- **Engagement Segments:** Based on loyalty programme participation
- **Channel Segments:** Online-first versus omnichannel customers

### Behavioural Segmentation
- **Shopping Pattern Groups:** Planned versus impulse shoppers
- **Price Sensitivity Clusters:** Value-conscious versus premium customers
- **Fashion Adoption Segments:** Trend leaders versus followers

### Value-Based Segments
- **High-Value Customers:** Top spending tier with high CLV
- **Growth Potential:** Medium spenders with increasing trends
- **Cost-Conscious:** Price-sensitive but loyal customer base

## 🛠️ Technical Implementation Considerations

### Data Requirements
- **Transaction Data:** Date, customer ID, article ID, price, channel
- **Customer Data:** Demographics, club membership, newsletter subscription
- **Article Data:** Category, seasonal classification, pricing information

### Feature Engineering Pipeline
1. **Data Integration:** Join transactions with customer and article metadata
2. **Aggregation:** Calculate customer-level metrics across time windows
3. **Categorisation:** Create categorical features from continuous variables
4. **Normalisation:** Standardise features for clustering algorithms
5. **Validation:** Ensure feature quality and business logic consistency

### Performance Optimisation
- Utilise Polars for efficient large-scale data processing
- Implement incremental feature updates for production systems
- Consider memory-efficient processing for multi-million record datasets
- Cache intermediate results for iterative analysis workflows

## 📋 Next Steps

1. **Feature Selection:** Choose appropriate feature subset based on business requirements
2. **Implementation:** Develop feature engineering pipeline using existing data infrastructure
3. **Validation:** Verify feature quality and business interpretation
4. **Segmentation:** Apply clustering algorithms to generated features
5. **Evaluation:** Assess segment quality using business metrics and statistical measures

## 📚 References

- Kumar, V., & Reinartz, W. (2016). *Creating Enduring Customer Value*. Journal of Marketing Research
- Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing
- Wedel, M., & Kamakura, W. A. (2000). *Market Segmentation: Conceptual and Methodological Foundations*. Kluwer Academic Publishers

---

*Document prepared for H&M Customer Data Analytics Project*  
*Last updated: 2025-08-06*