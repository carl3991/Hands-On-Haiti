#!/usr/bin/env python
# coding: utf-8

# <div style="
#     background: linear-gradient(90deg, #c8102e 0%, #002868 100%);
#     color: white;
#     padding: 18px 22px;
#     border-left: 6px solid #002868;
#     border-right: 6px solid #c8102e;
#     border-top: 3px solid #002868;
#     border-bottom: 3px solid #002868;
#     border-radius: 8px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.18);
#     font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
# ">
# 
#   <h1 style="
#       margin: 0 0 6px 0;
#       font-size: 30px;
#       font-weight: 700;
#       letter-spacing: 0.03em;
#       text-transform: uppercase;
#       color: white;
#   ">
#     Haiti Food Price Analysis
#   </h1>
# 
#   <p style="
#       margin: 0;
#       font-size: 15px;
#       opacity: 0.95;
#   ">
#     Exploring regional inequality, currency pressure, and structural fragility across Haiti’s food market economy.
#   </p>
# 
# </div>
# 

# <br></br>

# ### Project Description

# This project aims to understand trends in food affordability in Haiti by examining historical price data from 2005 to 2023, with a focus on identifying regional disparities such as the consistently higher prices in Artibonite and North‑West before 2015, and connecting these movements to national shocks like the COVID‑19 pandemic and worsening security conditions. The analysis matters because it reveals how localized challenges gradually evolved into a shared national crisis, where inflation, gang violence, and heavy dependence on imports amplified vulnerabilities across all regions. By blending statistical evidence with real‑world context, the notebook transforms raw data into a compelling narrative of economic fragility and resilience, showing how Haiti’s food markets reflect both regional inequalities and the broader pressures of global and domestic instability.

# <br></br>

# In[182]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # force integer type on xaxis

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[183]:


# Load dataset
file_id = "1VyLHX20ofimGB7jhFhhw06MBu9Ajod4o"
url = f"https://drive.google.com/uc?download&id={file_id}"
haiti_df = pd.read_csv(url)
print(haiti_df.head())


# In[184]:


# Show missing values
missing_values = haiti_df.isnull().sum()
print('The columns with missing values:\n', missing_values)


# In[185]:


# In case of existing duplicates
haiti_df.drop_duplicates(inplace=True)


# In[186]:


# Print data types
print(haiti_df.dtypes)


# In[187]:


# Convert date column to datetime
haiti_df['date'] = pd.to_datetime(haiti_df['date'])


# In[188]:


# Show data types
print(haiti_df.dtypes)


# 
# 
# ---
# 
# 

# # **Exploratory Analysis**

# In[189]:


# Glimpse on food prices statistics
haiti_df['usdprice'].describe().round(3)


# This summary of the price distribution tells a clear story about Haiti's food market dynamics:
# 
# * **Mean = 2.60, Median = 1.78:** The mean is noticeably higher than the median, which signals a right‑skewed distribution. Some very expensive commodities are pulling the average upward.
# 
# * **Standard deviation ≈ 2.59:** Prices in Haiti vary widely. That's a strong signal of inequality between cheap local staples and costly imports.
# 
# * **Range (0.06- 21.07):** Points to an enormous spread that highlights once again the gap between local goods and imported items.

# ### Histogram: Raw Price Distribution

# In[190]:


# Raw price histogram
plt.figure(figsize=(10, 6))
plt.hist(haiti_df['usdprice'], bins=20, color='black', edgecolor='lightgreen')
plt.xlabel('Price ($)', weight='bold', fontsize=14)
plt.ylabel('Frequency', fontsize=14, weight='bold')
plt.title('Price Distribution ', weight='bold', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# This histogram reveals a clear right-skewed pattern in Haiti's food price.
# Obviously, this **reflects economic inequality and market segmentation: basic staples are widely available and affordable, but imported items are out of reach for many, suggesting strong limited purchasing power.**

# ####Exchange rate effect?
# Because Haitians buy in gourdes but the dataset is converted to USD, local staples appear extremely cheap while imported goods remain expensive. Haiti’s chronic currency depreciation amplifies this gap, making the right‑skew more dramatic in USD terms than it would look in gourdes.

# **Log transformation Implementation will:**
# * Reduce skew and reveals a clearer price structure.
# * Make the distribution more balanced and easier to interpret.

# ### Histogram: Logged-price Distribution

# In[191]:


# Plot logged price histogram
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(haiti_df['usdprice']), bins=20, color='blue', edgecolor='red')
plt.xlabel('Logged Price ($)', weight='bold', fontsize=15)
plt.ylabel('Frequency', weight='bold', fontsize=15)
plt.title('Price Distribution ', weight='bold',fontsize=18)
# Remove top and right spines on plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# Despite remaining right skewed, it's much less extreme. The peak is now shifted slightly, and the tail is shorter, which means the data is less dominated by extreme values. Since the scale is now more uniform, it will easier to make comparisons between commodities or regions.
# 

# <br></br>

# ### **Food Measurement Systems: Haiti in Context**

# 
# In Haiti, food measurement systems rely heavily on traditional and informal units such as "marmite", "gode", "ti mammit", "boutey", and "sak", which are mostly volume‑based and vary by region or vendor. These units dominate street markets and informal trade, but they lack standardization, making packaging compatibility and pricing transparency extremely limited.
# 
# By contrast, the United States uses standardized weight and volume units like pounds, ounces, gallons, cups, and tablespoons, which are regulated by agencies such as the *USDA* and *FDA*. This system supports consistent packaging, clear price labeling, and easy integration with global trade, while Haiti’s reliance on traditional units complicates regulation, automation, and international alignment.
# <br></br>
# ##### Source:
# https://timothyschwartzhaiti.com/study-of-market-measuring-system-in-haiti/
# 

# <br></br>
# ##### Haiti's Food Measurements Description:
# * Marmite: A traditional Haitian volume unit, roughly equivalent to 1.5–2 liters, often used for grains like rice or maize. It's culturally embedded and widely used in markets.
# 
# * Pound: Standard weight unit (≈ 0.45 kg), used for imported or packaged goods.
# 
# * Gallon: Liquid volume unit (≈ 3.78 liters), typically used for oil or beverages.
# 
# * 350 G: Metric weight unit (≈ 0.77 pounds), likely used for small packaged items like sugar or flour.

# In[192]:


# TOtal count per unit
unit_count = haiti_df['unit'].value_counts()

# Plot with inner circle
plt.figure(figsize=(8, 8))
plt.pie(unit_count, labels= unit_count.index, autopct='%1.1f%%', startangle=290, pctdistance=0.72, colors= sns.color_palette('Spectral_r', len(unit_count)))
centre_circle= plt.Circle((0,0), radius=0.8, fc='white')
plt.gca().add_artist(centre_circle)
plt.title('Distribution of Units', weight='bold', fontsize=15)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ### Data shows:
# * The dominance of "Marmite", highlighting Haiti's reliance on traditional volume, while the presence of "Pound" and "Gallon" units reflects the influence of imported goods and liquid commodities.
# * The low frequency of "350 G" points that reflects limited packaging standardization in Haiti's local markets.

# <br></br>

# ### **Price Distribution vs Price Magnitude for Food Groups**

# In[193]:


# Create boxplot for Price Vs Food Type
plt.figure(figsize=(11,7))
sns.boxplot(x='food_type', y='usdprice', palette='Blues', fliersize=8, width=0.8, data=haiti_df)
plt.title('Price Outliers by Food Type', weight='bold',fontsize=17)
plt.xlabel('Food Type', weight='bold',fontsize=15)
plt.ylabel('Price ($)',weight='bold', fontsize=15)
plt.xticks(rotation=0)
plt.show()


# In[194]:


# Create kdeplot with price
plt.figure(figsize=(10,6))
sns.kdeplot(data=haiti_df, x='usdprice', hue='food_type', multiple='stack')
plt.title('Price Density by Food Type', weight='bold', fontsize=17)
plt.xlabel('Price ($)', weight='bold',fontsize=13)
plt.ylabel('Density', weight='bold', fontsize=13)
plt.tight_layout()
plt.show()


# #### Comment:
# * The Boxplot showed that oil and fats have a high median price, a wide interquartile range, and the most extreme outliers (some prices are above 20 dollars). This suggests that oil and fats are relatively scarce in Haiti's food economy, which contributes to their high prices and volatility.
# 
# * The KDE plot shows how frequently prices occur across the dataset. Cereals and tubers have a tall peak at low prices. This means they're very common and consistently cheap in Haiti. **This aligns with local consumption patterns, where rice is eaten widely and frequently, reinforcing its prominence in the food economy.**

# #### **Fun Fact!!!**
# Rice is loved in Haiti not just because it's cheap and filling, but because it's a cultural anchor, a symbol of resilience, and a daily ritual that unites families and communities; [A Grain of Unity](https://www.caribbeangreenliving.com/more-than-just-a-grain-exploring-the-shared-love-for-rice-in-haitian-culture/) if you will.

# <br></br>

# ### **Detecting Price Outliers**

# In[195]:


# Get commodities for Haiti
commodity_ordered= haiti_df['commodity'].value_counts().index.tolist()

# Create violin plot
plt.style.use('dark_background')
sns.set_context('paper')
plt.figure(figsize=(11,7))
sns.violinplot(y='commodity', x='usdprice', data=haiti_df,
               order=commodity_ordered, inner='quartile', density_norm='width', color='#BF00FF',  alpha=0.95)
plt.title('Price Distribution by Commodities', color='white', fontsize=25, fontweight='bold')
plt.xlabel('Price ($)',weight='bold', fontsize=19)
plt.ylabel('Commodity', weight='bold',fontsize=19)
plt.yticks(rotation=0, weight='bold', fontsize=10.5)
plt.xticks(weight='bold', fontsize=10.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(False)
plt.show()


# #### Comment:
# 1. Price volatility in imported commodities like wheat flour (imported) and oil (vegetable, imported) shows wide distributions, signaling **inconsistent pricing**. This tells us they're especially sensitive to phenomena like inflation, shipping costs, currency devaluation, or supply issues.
# 
# 2. Tighter clustering for local commodities Maize meal (local) and rice (tchako) display narrower violins, which is a sign of **price consistency**. This might be explained by tighter regulation in localized markets by the government.
# 
# 3. Right-skewed distribution is indicative of high inequality or sudden inflation spikes in certain regions.

# In[196]:


# Create boxplot for Price Vs Unit
sns.set_style('white')
sns.set_context('paper')
plt.figure(figsize=(11,7))
sns.boxplot(x='unit', y='usdprice', hue='unit', legend=False, palette='Reds', fliersize=8, width=0.8, data=haiti_df)
plt.title('Price by Unit', weight='bold',fontsize=17)
plt.xlabel('Unit', weight='bold',fontsize=15)
plt.ylabel('Price ($)', weight='bold',fontsize=15)
plt.show()


# In[197]:


# Create a catplot for Price per departments
sns.set_style('white')
sns.set_context('paper')
cat = sns.catplot(data=haiti_df, x='usdprice', col='Department', kind='box', showfliers=True, color='blue',col_wrap=3, height=3.6, aspect=1.1)
cat.set_titles("{col_name}", fontsize= 13, weight='bold')
cat.fig.suptitle('Food Prices Outliers by Department', fontweight= 'bold',fontsize=20)
cat.fig.subplots_adjust(top=0.90, hspace=1.1, bottom=0.075)

# To show Price ($) on each row
for ax in cat.axes.flatten():
    ax.set_xlabel('') # this is to remove the 'usdprice' as default
    ax.text(0.5, -0.2, "Price ($)", transform=ax.transAxes,
            ha='center', fontsize=10, weight='bold') # To include Price($) text
    ax.tick_params(axis='x', labelbottom=True) # For showing Price($) in every row


# <br></br>

# ### **Commodity Consumption and Pricing in Haiti**

# In[198]:


# Top 10 median price by commodities
median_prices = haiti_df.groupby('commodity')['usdprice'].median().sort_values(ascending=False).head(10)

# Create color palette
col = ['red' if i=='Oil (vegetable, imported)' else 'blue' for i in median_prices.index]

# Plot top 10 commodities median price
sns.set_style('white')
sns.set_context('paper')
plt.figure(figsize=(10,6))
sns.barplot(data=median_prices, palette=col, edgecolor='white')
plt.xlabel('Commodity', weight='bold',fontsize=14)
plt.ylabel('Median Price ($)', weight='bold', fontsize=14)
plt.title('Top 10 Most Expensive Commodities in Haiti', weight='bold',fontsize=18)
plt.xticks(rotation=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# #### Comment:
# 
# * The median price leader, *Oil (vegetables, imported)*, suggests that processed imports carry the highest financial burden. This aligns well with Haiti's dependency on imports, its vulnerability to global inflationary shocks, and exchange rate pressures.
# 
# * It is not surprising that beans and sugar (both consumed regularly) have elevated costs. It will likely impact low-income Haitians the hardest, especially in regions with limited agricultural productivity or limited market access.

# In[199]:


# Top 10 commodities frequency
top_frequent_commodities= haiti_df['commodity'].value_counts(normalize=True)*100
top_frequent_commodities= top_frequent_commodities.head(10).reset_index()
top_frequent_commodities.columns = ['commodity', 'frequency']

# Color palette
colors = ['lightgreen' if x=='Wheat flour (imported)' else 'lightgreen' if x=='Maize meal (local)' else 'grey' for x in top_frequent_commodities.index]

# Plot top common goods
sns.set_style('white')
sns.set_context('paper')
plt.figure(figsize=(10,6))
ax= sns.barplot(data=top_frequent_commodities, x='frequency', y='commodity', palette='viridis_r', edgecolor='lightblue')
for index, row in top_frequent_commodities.iterrows():
  ax.text(row['frequency'] + 0.5, index, f"{row['frequency']:.2f}%", va='center', weight='bold', fontsize=10)
plt.xlabel('Frequency', weight='bold', fontsize=13)
plt.ylabel('Commdities', weight='bold', fontsize=13)
plt.title(' Top Most Frequent Commodities in Haiti', weight='bold', fontsize=16)
plt.yticks(rotation=0)
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# #### Comment:
# This plot reflects how frequently each commodity appears in the dataset in %. In this case, *wheat flour(imported)* and *Maize meal (local)* rank highest, suggesting they're more consistently available across regions and widely consumed or traded. Their frequent reporting plays a crucial role in tracking inflationary patterns, making them key indicators in Haiti's food economy.

# <br></br>

# ### **Comparative Analysis of Median Prices Across Departments**

# In[200]:


# Median price by Department
dept_prices = haiti_df.groupby('Department')['usdprice'].median().sort_values(ascending=False)

# Barplot
dept_prices.plot(kind='bar', color='navy',edgecolor='red', linewidth=1.8, figsize=(10, 6))
plt.title('Median Prices by Department', weight='bold', fontsize=17)
plt.ylabel('Median Price ($)', weight='bold',fontsize=14)
plt.xlabel('Department', weight='bold',fontsize=14)
plt.xticks(rotation=0, fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# #### Comment:
# The median price variation across Haiti's departments is striking: while most departments are in the 1.8 – 2.0 dollar range, **Artibonite and Noth-West stand out significantly, exceeding 2.5 dollars**.  This price elevation may reflect limited road connectivity, strained local supply chains, and traffic jams caused by inadequate infrastructure.

# <br></br>

# #### **Local vs. Imported Insight across Departments**
# 

# In[201]:


# Filter for two commodities (local vs imported)
commodities = ['Wheat flour (imported)', 'Maize meal (local)']
filtered_df = haiti_df[haiti_df['commodity'].isin(commodities)]

# Group and pivot to get median prices by Department
# N.B. - the unstack() is to move 'commodity' from index to columns to make the side-by-side bar with unique colors possible
# N.B. - the reindex() is to make sure the columns are present in the exact order(Wheat flour and Maize meal) to be filled with the same color.
medians = filtered_df.groupby(['Department','commodity'])['usdprice'].median().unstack()
medians = medians.reindex(columns=commodities)

# Plot the side-by-side bar
medians.plot(kind='bar', figsize=(10, 6), color=['gold','red'], edgecolor='black', linewidth=1.2)
plt.title('Median Price Comparison: Wheat Flour (Imported) vs. Maize Meal (Local)', fontsize=14, weight='bold')
plt.xlabel('Department',weight='bold', fontsize=15)
plt.ylabel('Median Price ($)',weight='bold', fontsize=15)
# Remove top and right spines on plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(commodities)
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# A comparative look at maize meal (local) and wheat flour (imported) highlights a core structural pattern in Haiti's food economy: **imported commodities exhibit greater price volatility and regional disparity than their local counterparts**. For instance, wheat flour's median prices spike sharply in South-East, while maize meal shows more modest variation, with the North department as a notable outlier. These differences likely stem from supply chain complexity and *import dependency*, all of which disproportionately affect the pricing of foreign goods.

# Notably, the spread here isn't as extreme as with imported wheat flour (imported), but the fact that even a domestic commodity like maize meal (local) shows regional variability is telling. The spike may be reflecting regional inflationary pressures, especially because maize meal (local) is a core staple that shows up across most market listings. **This demonstrates how imported goods are not only more expensive but also more unstable in Haiti.** However, local commodities aren't immune to price fluctuations, as shown in the North department, likely due to poor road connectivity.

# 
# 
# ---
# 
# 

# <br></br>

# # **Time Series Analysis**

# In[202]:


# Create year column and fetch only years
haiti_df['year'] = haiti_df['date'].dt.year.astype(int)

# Group median prices by year
yearly_prices = haiti_df.groupby('year')['usdprice'].median().reset_index()

# Custom plot
sns.set_style('white')
sns.set_context('paper')

# Overall median price lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_prices, x='year', y='usdprice', linewidth=5.5, color='blue', marker='h',
markerfacecolor='white', markeredgecolor= 'white',markeredgewidth=3.2)
sns.lineplot(data=yearly_prices, x='year', y='usdprice', linewidth=2.5, color='red', marker='h')
plt.title("Haiti: Median Food Price Evolution", fontsize=16,weight='bold')
plt.ylabel("Median Prices ($)", weight='bold', fontsize=14)
plt.xlabel("Year", weight='bold', fontsize=14)

# Force integer ticks on x-axis
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Remove top and right spines on plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# ### **Median Food Prices in Haiti: Regional Trends Over Time**

# In[203]:


# Group median prices by department and year
dept_year = haiti_df.groupby(['Department', 'year'])['usdprice'].median().reset_index()

# Lineplot for each department
plt.figure(figsize=(10, 7))
sns.lineplot(data=dept_year, x='year', y='usdprice', hue='Department', linewidth=2)
plt.title("Median Prices Evolution by Department", weight='bold', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Year", weight='bold', fontsize=14)
plt.ylabel("Median Prices ($)", weight='bold', fontsize=14)
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=10,integer=True)) # ensure xaxis shows integer
# Remove top and right spines on plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# #### Comment:
# The convergence and collective rise post-2019 hints at a **nationwide pressure on food affordability**, not just isolated spikes.

# <br></br>

# ### **Security Issues as a Driver?**
# Armed groups have expanded their reach, blocking roads, ports, and fuel supplies. This has disrupted food distribution and raised transport costs. Regions like Artibonite and North‑West, already facing supply constraints, were hit harder when insecurity restricted movement. By 2025, [over 5.7 million Haitians faced acute food insecurity](https://news.un.org/en/story/2025/10/1166080
# ), with nearly 2 million at emergency levels.
# 
# 
# ### **COVID‑19 Pandemic Effects?**
# * **Lockdowns** (*“Peyi Lock”*): Movement restrictions reduced economic activity, cut incomes, and limited access to markets.
# 
# * **Import dependence:** Data supported that Haiti relies heavily on imported staples. Global trade disruptions and currency depreciation during COVID‑19 made food more expensive.
# 
# * **Price spikes:** The [World Food Programme](https://www.wfp.org/stories/haiti-coronavirus-high-food-prices-and-how-beans-became-luxury) reported beans and other basics became “luxury items” as pandemic‑related inflation hit households.
# 
# 
# The post‑2019 sharp rise across all departments aligns with nationwide shocks: **first COVID‑19, then escalating insecurity**. Price evolution across department demonstratedd how, already severe localized problems, gave way to a shared national crisis.

# Sources:
# 
# 
# https://hopeforhaitians.org/covid-19-impact-on-food-security-in-haiti/

# <br></br>

# # **Regional Price Forecast Comparison: Artibonite, Noth-West, and West**

# This section compares forecasted food price trends across three Haitian departments using ARIMA models tailored to each region's volatility and structural behavior.

# <br></br>

# **Artibonite Price Forecast**

# In[204]:


# Filter Artibonite series
art_series = dept_year[dept_year['Department'] == 'Artibonite'].set_index('year')['usdprice']

# Log transform + Difference 2nde order (because variance was unstable)
art_log = np.log1p(art_series)
art_log_diff2 = art_log.diff().diff().dropna()

adf_result_log2 = adfuller(art_log_diff2)
print("ADF Statistic (log + diff2):", adf_result_log2[0])
print("p-value:", adf_result_log2[1])

# Fit ARIMA with d = 2
model_art = ARIMA(art_log, order=(1, 2, 1))
fit_art = model_art.fit()

print(fit_art.summary())

# Forecast next 5 years
forecast_art = fit_art.get_forecast(steps=5)
forecast_ci_art = forecast_art.conf_int()


# #### **Result Interpretation for Artibonite**
# #### Coefficients:
# * AR(1) coefficient = -0.50, p = 0.026: Strong and statistically significant autoregressive effect. Past values strongly influence current ones.
# 
# * MA(1) coefficient = −0.74, p = 0.057: Borderline significant and not as strong as the AR term. Useful to smooth irregulrities in the differenced series.
# 
# * sigma = 0.0122(p=0.010): Residual is small and statistically meaningful, thus the model captures most of the structure in the data.
# 
# #### Diagnostics:
# * Ljung-Box Q (p = 0.82): Residuals show no significant autocorrelation, signaling that the model is not missing obvious patterns.
# 
# * Jarque-Bera (p = 0.84): Residuals are approximately normally distributed.
# 
# * Heteroskedasticity (p = 0.19): Residuals are stable.
# 
# * Skew = 0.35, Kurtosis = 3.19: Residuals are slighly skew to the right and close to normal distribution (ideal is skew ≈ 0, kurtosis ≈ 3).
# <br></br>
# ### What it Means for Artibonite Forecast:
# **The ARIMA(1,2,1) model is adequate: no residuals autocorrelation, AR term is strongly significant, MA term is barely significant, and diagnostics are clean (residuals are normally distributed). This suggest that sudden shocks are not the dominant forces in Artibonite region. Instead, the forecast mirrors the inflationary and import-dependent pressure that defines its natural market trend (growing inflation, currency depreciation, and dependence on imported goods).**

# In[205]:


observed = art_series

# Back-transform forecast from log to USD
forecast_values = np.expm1(forecast_art.predicted_mean)
forecast_lower = np.expm1(forecast_ci_art.iloc[:, 0])
forecast_upper = np.expm1(forecast_ci_art.iloc[:, 1])

# Create forecast horizon years
last_year = observed.index[-1]
future_years = range(last_year + 1, last_year + 6)

# Plot
plt.figure(figsize=(10,6))
plt.plot(observed.index, observed, label="Observed", color="blue", linewidth=2) # observed series
plt.plot(future_years, forecast_values, label="Forecast", color="red", linewidth=2) # forecast line
# Confidence interval
plt.fill_between(future_years, forecast_lower, forecast_upper,
                 color="pink", alpha=0.3)

plt.title("ARIMA Forecast for Artibonite", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14, weight='bold')
plt.ylabel("Median Price ($)", fontsize=14, weight='bold')
plt.legend()
plt.grid(False)
plt.show()


# <br></br>

# **North-West Price Forecast**

# In[206]:


# Filter north-west series
nw_series = dept_year[dept_year['Department'] == 'North-West'].set_index('year')['usdprice']
nw_log = np.log1p(nw_series)  # log(1+x) to avoid log(0)
nw_log_diff = nw_log.diff().diff().diff().dropna()

# Check stationarity
adf_result = adfuller(nw_log_diff)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# Fit ARIMA
model_nw = ARIMA(nw_log, order=(1,2,1))
fit_nw = model_nw.fit()
print(fit_nw.summary())

# Forecast next 5 years
forecast_nw = fit_nw.get_forecast(steps=5)
forecast_ci_nw = forecast_nw.conf_int()


# #### **Result Interpretation for North‑West**
# #### Coefficients:
# * AR(1) = -0.1127 (p = 0.730) suggest no autoregressive effect, indicating that past values of the series does not predict future values.
# 
# * MA(1) = -0.7992 (p = 0.047): Significant, thus capturing short-term shocks well.
# 
# * Sigma2 = 0.0629 (p = 0.054): Borderline significant, as the model captures most of the structure but some noise remains.
# 
# ### Diagnostics:
# * Ljung‑Box Q (p = 0.74): Residuals show no autocorrelation.
# 
# * Jarque‑Bera (p = 0.83):  Residuals are normally distributed.
# 
# * Heteroskedasticity (p = 0.27): Residuals are borderline stable, suggesting slight variance instability.
# 
# * Skew = −0.37, Kurtosis = 3.04: Residuals are close to normal shape.
# <br></br>
# ### What it Means for North-West Forecast:
# **The ARIMA(1,2,1) provides the best balance of statistical fit and economic interpretability for the North‑West region. The significant MA(1) term indicates that North-West is dominated by sudden shocks rather than market trend, meaning that external shocks (insecurity, road blockages, supply disruptions, and other structural constraints) create abrupt price movements that this model captures well. In other words, instead of following a smooth, predictable trajectory, North-West prices respond to short-term disruptions, reflecting the region's fragile market conditions and chronic volatility.**

# In[207]:


forecast_values_nw = np.expm1(forecast_nw.predicted_mean)
forecast_lower_nw = np.expm1(forecast_ci_nw.iloc[:, 0])
forecast_upper_nw = np.expm1(forecast_ci_nw.iloc[:, 1])

observed_nw = nw_series
last_year_nw = observed_nw.index[-1]
future_years_nw = range(last_year_nw + 1, last_year_nw + 6)

plt.figure(figsize=(10,6))

plt.plot(observed_nw.index, observed_nw, label="Observed", color="blue",linewidth=2)
plt.plot(future_years_nw, forecast_values_nw, label="Forecast", color="red", linewidth=2)

plt.fill_between(future_years_nw, forecast_lower_nw, forecast_upper_nw,
                 color="pink", alpha=0.3)

plt.title("ARIMA Forecast for North-West", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14, weight='bold')
plt.ylabel("Median Price ($)", fontsize=14, weight='bold')
plt.legend()
plt.grid(False)
plt.show()


# <br></br>

# **West Price Forecast**
# 

# Because West’s price series reflects structural stagnation rather than trend-driven inflation, ARIMA(1,1,2) provides the most stable and interpretable model, even though the ADF test does not indicate stationarity after multiple differences.

# In[208]:


# Filter West series
west_series = dept_year[dept_year['Department'] == 'West'].set_index('year')['usdprice']
west_log = np.log1p(west_series)  # log(1+x) to avoid log(0
west_diff = west_log.diff().dropna()
# Check stationarity
adf_result = adfuller(west_diff)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# Fit ARIMA (same order (1,1,1))
model_west = ARIMA(west_log, order=(1,1,2))
fit_west = model_west.fit()
print(fit_west.summary())

# Forecast next 5 years
forecast_west = fit_west.get_forecast(steps=5)
forecast_ci_west = forecast_west.conf_int()


# #### **Result Interpretation for West**
# 
# #### Coefficients:
# * AR(1) = 0.9757 (p = 0.001) suggest a highly significant autoregressive effect. Past values strongly influence current ones.
# 
# * MA(1) and MA(2) = −0.88 (both p = >0.05): The Moving Average is not statistically significant, so it doesn't add much explanatory power.
# 
# * Sigma2 = 0.0111 (p = 0.887). The model doesn't capture volatility well.
# 
# ### Diagnostics:
# * Ljung‑Box Q (p = 0.87): Residuals show no strong autocorrelation, but it's close to the 0.05 threshold. The model is mostly adequate, but worth checking higher lags.
# 
# * Jarque‑Bera (p = 0.02): Residuals are not normally distributed, likely because of occasional price shocks in the West.
# 
# * Heteroskedasticity (p = 0.51): Residuals are stable.
# 
# * Skew = 1.34, Kurtosis = 4.95: Residuals are rihgt-skewed and  heavily-tailed. This indicateshigh price asymmetry, likely with the consistent insecurity crisis and supply disruptions.
# <br></br>
# 
# ### What it means for West Forecast:
# **The ARIMA(1,1,2) model for West shows that prices barely move, reflecting a market where normal inflationary forces are overshadowed by chronic instability. Instead of following a steady upward trend, prices remain unusually flat because persistent insecurity, blocked transport routes, and repeated disruptions suppress both supply and demand. Overall, the West behaves like a paralyzed market, shaped by more instability and disruption than by predictable economic forces.**

# In[209]:


forecast_values_west = np.expm1(forecast_west.predicted_mean)
forecast_lower_west = np.expm1(forecast_ci_west.iloc[:, 0])
forecast_upper_west = np.expm1(forecast_ci_west.iloc[:, 1])
observed_west = west_series
last_year_west = observed_west.index[-1]
future_years_west = range(last_year_west + 1, last_year_west + 6)

# Plot West forecast
plt.figure(figsize=(10,6))
plt.plot(observed_west.index, observed_west, label="Observed", color="blue", linewidth=2)
plt.plot(future_years_west, forecast_values_west, label="Forecast", color="red", linewidth=2)
plt.fill_between(future_years_west, forecast_lower_west, forecast_upper_west,
                 color="pink", alpha=0.3)

plt.title("ARIMA Forecast for West", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14, weight='bold')
plt.ylabel("Median Price ($)", fontsize=14, weight='bold')
plt.legend()
plt.grid(False)
plt.show()


# <br></br>
# ## **Directional Forecast Summary**
# #### **Artibonite:**  
# The forecast rises steadily, extending its long‑run trend. Prices keep climbing slowly but persistently, reflecting ongoing inflation and import dependence.
# 
# #### **North‑West:**
# The forecast also moves upward but with more volatility. Wider confidence bands reflect a shock‑driven region shaped by instability and supply disruptions.
# 
# #### **West:**
# The forecast stays almost flat. This stagnation signals a paralyzed market where chronic insecurity and disrupted supply chains suppress normal price movement.

# #

# ## Why the West's Price Prediction Might be Worse?

# Price stagnation in Haiti’s West department does not signal stability but rather severe market paralysis. In Port‑au‑Prince, armed groups disrupt transport and market access, cutting off the flow of goods and creating artificial scarcity. Combined with high national inflation and heavy reliance on imports, households face rising costs without local production to cushion the blow. At the same time, widespread displacement and acute food insecurity suppress demand, while insecurity prevents trucks from reaching markets and breaks rural‑urban supply chains. Together, these pressures explain why prices in the West appear flat, but actually is a **reflection of economic dysfunction and suppressed market activity** rather than equilibrium.

# Sources:
# 1. [IPC Acute Food Insecurity Snapshot](https://www.ipcinfo.org/fileadmin/user_upload/ipcinfo/docs/IPC_Haiti_Acute_Food_Insecurity_Mar_Jun2025_Snapshot_English.pdf)
# 2. [Fews NET Food Security Outlook](https://fews.net/latin-america-and-caribbean/haiti/food-security-outlook/october-2024)
# 3. [FAO GIEWS Country Brief](https://www.fao.org/giews/countrybrief/country.jsp?code=HTI)
# 4. [UN News – Record hunger in Haiti](https://news.un.org/en/story/2025/04/1162391)

# <br></br>

# ### **Model Diagnostic**

# #### For Artibonite

# In[210]:


# Residuals from ARIMA model
residuals = fit_art.resid

#  Histogram of residuals
plt.figure(figsize=(10,4))
sns.histplot(residuals, kde=True, color="red")
plt.title("Residual Distribution")
plt.show()


# In[211]:


# Safe lag selection for ACF and PACF
max_lags_acf = min(20, len(residuals) - 1)          # ACF can go up to n-1
max_lags_pacf = min(len(residuals) // 2 - 1, 20)    # PACF must be < 50% of sample size

# Autocorrelation (ACF) plot
plot_acf(residuals, lags=max_lags_acf)
plt.show()


# In[212]:


# Partial Auto Correlation (PACF) plot
plot_pacf(residuals, lags=max_lags_pacf)
plt.show()

# Ljung-Box test for autocorrelation
ljung_box = acorr_ljungbox(residuals, lags=[min(10, len(residuals) - 1)], return_df=True)
print()
print(ljung_box)


# ### Interpretation:
# The p-value is very high (>0.05). This  means that ARIMA residuals do not show significant autocorrelation, suggesting that the model captures the structure of the data reasonably well.

# <br></br>

# #### For West

# In[213]:


# Residuals from ARIMA model
residuals = fit_west.resid

#  Histogram of residuals
plt.figure(figsize=(10,4))
sns.histplot(residuals, kde=True, color="orange")
plt.title("Residual Distribution")
plt.show()


# In[214]:


# Safe lag selection for ACF and PACF
max_lags_acf = min(20, len(residuals) - 1)          # ACF can go up to n-1
max_lags_pacf = min(len(residuals) // 2 - 1, 20)    # PACF must be < 50% of sample size

# Autocorrelation (ACF) plot
plot_acf(residuals, lags=max_lags_acf)
plt.show()


# In[215]:


# Partial Auto Correlation (PACF) plot
plot_pacf(residuals, lags=max_lags_pacf)
plt.show()

# Ljung-Box test for autocorrelation
ljung_box = acorr_ljungbox(residuals, lags=[min(10, len(residuals) - 1)], return_df=True)
print()
print(ljung_box)


# ### Interpretation:
# Since p > 0.05, we fail to reject the null that residuals do not show significant autocorrelation.
# This is a good sign because the ARIMA model for West is capturing the structure well, and the residuals behave like random noise.

# <br></br>

# #### For North-West

# In[216]:


# Residuals from ARIMA model
residuals = fit_nw.resid

#  Histogram of residuals
plt.figure(figsize=(10,4))
sns.histplot(residuals, kde=True, color="pink")
plt.title("Residual Distribution")
plt.show()


# In[217]:


# Safe lag selection for ACF and PACF
max_lags_acf = min(20, len(residuals) - 1)
max_lags_pacf = min(len(residuals) // 2 - 1, 20)    # PACF must be < 50% of sample size

# Autocorrelation (ACF) plot
plot_acf(residuals, lags=max_lags_acf)
plt.show()


# In[218]:


# Partial Auto Correlation (PACF) plot
plot_pacf(residuals, lags=max_lags_pacf)
plt.show()

# Ljung-Box test for autocorrelation
ljung_box = acorr_ljungbox(residuals, lags=[min(10, len(residuals) - 1)], return_df=True)
print()
print(ljung_box)


# ### Interpretation
# 
# Since p > 0.05, we fail to reject the null that the residuals show no significant autocorrelation. Similar to the other department's residuals, the ARIMA model for North‑West has captured the structure well, leaving residuals that behave like random noise.

# 
# 
# ---
# 
# 

# # **Findings & Discussion**
# 

# Haiti's food prices show a clear pattern of inequality, import independence, and structural fragility. Prices appear cheap in the dataset because they are expressed in USD. In practice, Haitians pay in *gourdes*, and as the gourde depreciates, even “low” USD prices become increasingly expensive for local households. This means the right‑skewed distribution doesn't just reflect a few costly imports: it also hides the growing burden of everyday staples as the exchange rate widens. What looks affordable on paper can feel unaffordable in real life, especially for families whose income doesn't keep pace with currency depreciation. Traditional units like marmite dominate local markets, while imported items (often sold in pounds or gallons) are both pricier and more volatile. Local staples such as maize meal and rice show stable, tightly clustered prices, whereas imported wheat flour and vegetable oil display wide swings driven by inflation, shipping costs, and currency depreciation. Regional differences add another layer: while most departments fall near the same median price, places like Artibonite and North‑West face higher costs due to weak infrastructure and supply constraints. After 2019, however, these regional gaps begin to narrow, not because conditions improved, but because nationwide shocks like COVID‑19 and rising insecurity pushed prices upward everywhere, creating a shared pressure on food affordability across the country.
# <br></br>
# The forecasts highlight how these pressures play out differently across regions. Artibonite continues along a steady, predictable upward path, reflecting a trend‑driven market shaped by persistent inflation and import dependence. North‑West also rises but with greater volatility, as the model captures a shock‑driven environment marked by insecurity, fragile supply chains, and frequent disruptions. West, however, stands in sharp contrast: its forecast remains almost flat, signaling a market where chronic instability, blocked transport routes, and disrupted economic activity suppress normal price movement. Taken together, the three forecasts show that while Haiti's regions face the same national shocks, their price dynamics diverge sharply depending on local conditions and structural vulnerabilities.

# <br></br>

# ### Presented by:
# 
# <a href="https://www.linkedin.com/in/carllegros/" target="_blank" style="color:blue;">Carl Legros </a>
