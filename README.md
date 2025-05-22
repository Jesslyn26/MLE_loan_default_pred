# Loan Default Prediction

A data preprocessing pipeline that is build to automatically clean and process data from different resources (loan daily, clickstream, attributes, and financial) to predict if the loan should be defaulted or not. this will help to improve the company's decision on which user to give loans or not

# Dataset description
## Features
* Attributes: customers' personal information 
* Financial: customers's 
* Clickstream: user clickstream information across the span of 2 years. this can help us to understand user's pattern when they 



## Data Pipeline

1. **Bronze Table Creation**:
   - Processes raw datasets (clickstream, attributes, financials).
   - Cleans and filters data by snapshot date.
   - Saves processed data in the `datamart/bronze/` directory.

2. **Silver Table Creation**:
   - Enhances Bronze tables with additional transformations.
   - Saves processed data in the `datamart/silver/` directory.

3. **Gold Table Creation**:
   - Combines Silver tables into a unified Gold table.
   - Performs feature engineering (e.g., one-hot encoding, scaling).
   - Saves processed data in the `datamart/gold/` directory.

4. **Data Cleaning and Preprocessing**:
   - Handles missing values, outliers, and invalid data.
   - Imputes missing values using mean/median.
   - Normalizes and encodes categorical features.

5. **Visualization and Insights**:
   - Generates correlation matrices and heatmaps.
   - Provides insights into data distributions and relationships.
