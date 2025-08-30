# Behavioural Segmentation Features for H&M Dataset

This document outlines the specific features required for behavioural customer segmentation based on the actual fields available in the H&M dataset.

## Dataset Structure Overview

### Customer Dataset (7 columns)
- **`customer_id`** (String) - Unique anonymised customer identifier
- **`FN`** (Float64) - Unknown field (57.31% missing values)
- **`Active`** (Float64) - Unknown field (57.95% missing values)
- **`club_member_status`** (String) - Membership tier (ACTIVE, PRE-CREATE, etc.)
- **`fashion_news_frequency`** (String) - Email engagement (Regularly, Monthly, NONE)
- **`age`** (Int64) - Customer age
- **`postal_code`** (String) - Anonymised postal code for geographic clustering

### Transactions Dataset (5 columns)
- **`t_dat`** (String) - Transaction date (2018-09-20 to 2020-09-22)
- **`customer_id`** (String) - Links to customer data
- **`article_id`** (Int64) - Links to product data
- **`price`** (Float64) - Anonymised price (preserves relative spending patterns)
- **`sales_channel_id`** (Int64) - Channel (1=online, 2=store)

### Articles Dataset (25 columns)

#### Product Identification
- **`article_id`** - Unique identifier for each article
- **`product_code`** - Product code identifier

#### Product Description
- **`prod_name`** - Product name
- **`detail_desc`** - Detailed product description

#### Product Type Classification
- **`product_type_no`** - Product type number
- **`product_type_name`** - Product type name (e.g., "Sunglasses", "Shorts", "Trousers")
- **`product_group_name`** - Product group classification

#### Visual Appearance
- **`graphical_appearance_no`** - Graphical appearance number
- **`graphical_appearance_name`** - Graphical appearance name

#### Colour Information
- **`colour_group_code`** - Colour group code
- **`colour_group_name`** - Colour group name
- **`perceived_colour_value_id`** - Perceived colour value ID
- **`perceived_colour_value_name`** - Perceived colour value name
- **`perceived_colour_master_id`** - Perceived colour master ID
- **`perceived_colour_master_name`** - Perceived colour master name

#### Department and Organisation
- **`department_no`** - Department number
- **`department_name`** - Department name

#### Index Classification
- **`index_code`** - Index code
- **`index_name`** - Index name
- **`index_group_no`** - Index group number
- **`index_group_name`** - Index group name

#### Section and Garment Classification
- **`section_no`** - Section number
- **`section_name`** - Section name
- **`garment_group_no`** - Garment group number
- **`garment_group_name`** - Garment group name

## Core Behavioural Features to Engineer

### RFM Analysis Features

#### Recency Features
- **Days since last purchase** - Calculated from `t_dat`
- **Purchase recency percentile** - Customer's recency rank relative to all customers
- **Recency trend** - Whether customer is becoming more or less active

#### Frequency Features
- **Total transaction count** - Number of purchases per customer
- **Purchase frequency per month** - Average monthly transaction rate
- **Shopping consistency** - Standard deviation of time between purchases
- **Customer lifespan** - Days between first and last purchase

#### Monetary Features
- **Total spend** - Sum of all `price` values per customer
- **Average transaction value** - Mean spend per purchase
- **Spending volatility** - Standard deviation of transaction values
- **Price tier preference** - Average price percentile within categories

### Purchase Pattern Features

#### Temporal Behaviour
- **Weekend vs weekday preference** - Shopping pattern ratios
- **Seasonal preference** - Peak shopping quarters/months
- **Purchase timing consistency** - Regularity of shopping intervals between days

#### Same-Day Purchase Patterns
- **Daily purchase volume** - Items purchased per day by customer
- **Cross-category shopping frequency** - Purchases across different departments on same day
- **Daily spending patterns** - Spending distribution across shopping days

### Product Preference Features

#### Category Behaviour
- **Product type diversity** - Number of unique `product_type_name` purchased
- **Department spread** - Number of different `department_name` shopped
- **Garment group preferences** - Primary garment categories purchased
- **Category concentration** - Percentage of purchases in top category

#### Style Preferences
- **Colour preference scores** - Top 3-5 `colour_group_name` frequencies
- **Pattern preference** - Solid vs patterned item ratios (from `graphical_appearance_name`)
- **Colour intensity preference** - Light, medium, or dark colour preferences
- **Style consistency score** - How consistent customer's style choices are

### Channel & Engagement Features

#### Shopping Channel Behaviour
- **Channel preference ratio** - Online vs store preference (from `sales_channel_id`)
- **Multi-channel behaviour** - Usage of both online and store channels
- **Channel switching patterns** - Frequency of changing between channels

#### Customer Engagement
- **Fashion engagement level** - Encoded `fashion_news_frequency`
- **Loyalty tier status** - `club_member_status` categories
- **Engagement consistency** - Stability of engagement over time

### Advanced Behavioural Features

#### Price Sensitivity
- **Premium vs budget preference** - Tendency towards higher/lower priced items
- **Price variation response** - Response to price changes within product categories (limited by anonymised pricing)

#### Product Adoption Patterns
- **Popular product alignment** - Purchasing products that are popular across customer base
- **Seasonal shopping patterns** - Timing of purchases across different seasons

#### Product Discovery
- **Product type exploration** - Frequency of trying different product categories
- **Product family loyalty** - Repeat purchases within same product codes/families
- **Product exploration breadth** - Range of different product types and categories tried

### Text-Based Features (Advanced)

#### Product Description Analysis
- **Style keywords** - TF-IDF features from `detail_desc`
- **Product description similarity** - Clustering based on description preferences
- **Feature preference extraction** - Key product features mentioned in descriptions

## Feature Engineering Priorities

### Phase 1: Core RFM Features
1. Recency, Frequency, Monetary calculations
2. Basic purchase patterns (seasonality, channel preference)
3. Product category preferences

### Phase 2: Advanced Behavioural Features
1. Price sensitivity and trend adoption
2. Style consistency and preference scores
3. Cross-category shopping patterns

### Phase 3: Text and Advanced Analytics
1. Product description NLP features
2. Complex behavioural pattern recognition
3. Predictive behavioural indicators

## Dataset Quality for Segmentation

- **Total customers**: 1,371,980 unique customers
- **Total transactions**: 31+ million transaction records
- **Date range**: September 2018 to September 2020 (2+ years of data)
- **Product catalogue**: 105,542 unique articles
- **Missing data**: Minimal in key behavioural fields (<1% in most cases)

The anonymised pricing preserves relative spending patterns whilst protecting competitive information, making this dataset exceptionally well-suited for sophisticated behavioural segmentation analysis.