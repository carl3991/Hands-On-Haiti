# Hands-On Haiti: Food Price Modeling & Geospatial Insights

This project analyzes food price dynamics in Haiti using a full data‑science workflow that blends exploratory data analysis, time‑series exploration, machine‑learning modeling, and geospatial visualization. The goal is to understand how staple food prices evolve across regions and time, identify the strongest predictive models, and communicate insights in a way that is accessible to both technical and non‑technical audiences.

## Exploratory Data Analysis (EDA)
The analysis begins with a structured EDA to understand the distribution, variability, and seasonality of unit measurements and key food items such as rice, beans, maize, and cooking oil.

Key steps include:

1. Cleaning and standardizing price data across markets and departments

2. Visualizing distributions using boxplots, KDE plots, and histograms

3. Identifying outliers and regional disparities

4. Examining correlations between commodities and market locations



## Time Series Analysis
To understand how prices evolve over time, the project includes a full time‑series exploration:

1. Trend, seasonality, and noise decomposition
2. Rolling averages and volatility analysis
3. Stationarity testing (ADF)
4. Autocorrelation and partial autocorrelation diagnostics

These diagnostics guide model selection and help determine whether prices follow predictable seasonal cycles or respond more to shocks and structural changes.

## Machine Learning Models
Multiple models were tested to predict future food prices and evaluate which approach best captures Haiti's market behavior.

**Models attempted:**

1. Multiple Linear Regression
2. Gradient Boost
3. Random Forest Regressor
4. ARIMA / SARIMA (time‑series)
5. XGBoost Regressor

**Best-performing model:** XGBoost Regressor delivered the strongest performance based on RMSE, and R-squared. The model handled non‑linear relationships and regional variability more effectively than the other algorithms.


## Geospatial Analysis with Folium
To visualize how food prices vary across Haiti’s department, the project integrates Folium for interactive mapping.

Geospatial components include:

* Mapping markets and departments
* Choropleth maps showing average prices by region
* Marker clusters for individual market observations

In addition to mapping median food prices by department (blue circle markers), I incorporated commodity weight information as a secondary marker layer. These weights appear as red circle markers nested inside the departmental clusters, highlighting the relative importance of key staples. 
To keep the visualization focused, weight markers were added only for the two most expensive departments, and for two staple commodities in each department whose weights represented more than one‑third of the total commodity basket. These maps help reveal spatial inequalities, regional hotspots, and potential logistical and economic drivers behind price differences.

## Project Outcomes
This project provides:

* A reproducible, well‑documented workflow

* Clear visualizations for both technical and public audiences

* A comparison of modeling approaches with a justified best model

* Cultural and economic context to ground the findings

* Interactive geospatial tools to support policy and humanitarian decision‑making
