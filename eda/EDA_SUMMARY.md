# Zara Sales EDA - Summary

**Dataset:** 20,252 unique products with no duplicates ✓

## Data Overview
- **Price Range:** $12.00 - $134.99 (Mean: $41.95, Median: $35.95)
- **Sales Volume:** 518 - 1,940 units (Mean: 1,097.4, Median: 990)
- **Price Concentration:** 53% of products between $11.88-$36.60
- **Product Terms:** 5 categories (jackets, jeans, shoes, sweaters, t-shirts)
- **Gender:** Balanced between MAN and WOMAN sections
- **Seasons:** 4 seasons covered (Winter, Spring, Summer, Autumn)
- **Materials:** 11 types (Polyester, Cotton, Wool, etc.)
- **Positions:** 3 types (Aisle, End-cap, Front of Store)

## Key Findings
- No duplicates, clean dataset
- Price is right-skewed with outliers on high end
- Sales volume concentrated in narrow range (IQR: 515.25 units)
- All product categories present across all price ranges

## Suggested Next Steps

### 1. Revenue Analysis
Create `revenue = sales_volume × price` column
- Identifies true business value (volume alone is misleading)
- Compare revenue by position, promotion, category, season

### 2. Product Position Impact
Analyze how placement affects performance:
- Does "Front of Store" drive higher sales?
- Does "End-cap" command premium pricing?
- Interaction with promotional status?

### 3. Correlation Analysis
Encode categorical variables and build correlation matrix:
```python
eda_df['is_promoted'] = (eda_df['promotion'] == 'Yes').astype(int)
eda_df['is_seasonal'] = (eda_df['seasonal'] == 'Yes').astype(int)
eda_df['is_front_of_store'] = (eda_df['product_position'] == 'Front of Store').astype(int)
```

Key relationships to explore:
- Price vs. Sales Volume (demand elasticity)
- Revenue vs. Position (strategic impact)
- Promotion effectiveness (lift on volume)
- Material correlation with price/sales
