# Hands On Haiti Food Economy: Staple Price Modeling & Geospatial Insights
<br></br>
## 1. Problem Statement
Haiti’s food markets have faced persistent instability driven by inflation, import dependence, and regional inequality. This project analyzes nearly two decades of food price data (2005–2023) to understand how affordability has changed over time and why certain departments, such as Artibonite and North West, consistently experience higher prices. The goal is to uncover the structural and shock-driven forces shaping food access across the country.
## 2. Why This Matters
Food affordability is central to household well-being, yet Haiti’s markets are highly vulnerable to global price swings, currency depreciation, and domestic insecurity. By examining long term trends and regional disparities, this project highlights how localized challenges have evolved into a nationwide crisis. The analysis blends statistical evidence with real-world context to support more informed policy, humanitarian planning, and economic decision-making.
## 3. Data Source
The project uses historical food price data collected across Haiti’s ten departments from 2005 to 2023. The dataset was obtained from Kaggle, but the original source is the World Food Programme (WFP), which conducts regular market monitoring across the country.
The dataset includes:
*	Market level observations
*	Unit measurements and commodity types
*	Prices ($) for staples such as rice, beans, maize, and cooking oil

## 4. Modeling Approach

### Exploratory Data Analysis (EDA)
*	Cleaning the date column and extracting year and month for improved temporal analysis
*	Visualizing distributions (boxplots, KDE plots, histograms)
*	Identifying outliers and regional disparities
### Time Series Analysis
*	Trend, seasonality, and residual decomposition
*	Rolling averages and volatility analysis
*	Stationarity testing (ADF)
*	ACF/PACF diagnostics for model selection
### Machine Learning Models
Models tested for price prediction include:
*	Multiple Linear Regression (Logged Price)
*	Gradient Boost
*	Random Forest Regressor
*	ARIMA / SARIMA
*	XGBoost Regressor
**Best model: XGBoost Regressor, based on RMSE and R-square, capturing non linear relationships and regional variability most effectively.**
### Geospatial Analysis (Folium)
*	Department level choropleth maps
*	Market level marker clusters
*	Commodity weight overlays for high cost regions

## 5. Key Insights
*	Prices show a strong upward trend, with clear structural inflation over time.
*	Artibonite and North West consistently rank among the most expensive regions.
*	After 2019, national shocks (COVID 19 and rising insecurity) pushed prices upward across all departments.
*	Weight heavy commodities reveal which staples contribute most to overall price pressure in the regions with the highest costs.
*	XGBoost outperformed other models in capturing non linear and region specific patterns.

## 6. Model Performance
*	XGBoost Regressor delivered the best predictive accuracy.
*	Random Forest performed well but showed slightly higher variance.
*	Linear Regression struggled with non linear relationships and regional heterogeneity.
*	Time series diagnostics confirmed strong trend components and irregular shocks across departments.

## 7. Limitations & Next Steps
*	Incorporating transportation routes, road quality, or distance to ports could improve modeling.
*	Future work may include more external variables (exchange rate and import volumes)

