Missing Value Handling Summary:

- t_dat: Interpolate based on customer patterns
- customer_id: Drop rows
- article_id: Drop rows
- price: Fill with median price for that article_id
- sales_channel_id: Fill with mode
- FN: Fill with 0
- Active: Fill with "UNKNOWN"
- club_member_status: Fill with "INACTIVE"
- fashion_news_frequency: Fill with "NONE"
- age: Fill with median age
- postal_code: Fill with "UNKNOWN"
- product_code: Fill with "UNKNOWN"
- prod_name: Fill with "UNKNOWN"
- Numerical codes: Fill with 0
- Categorical names: Fill with "UNKNOWN"
- detail_desc: Fill with "NO_DESCRIPTION"
