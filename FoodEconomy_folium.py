#!/usr/bin/env python
# coding: utf-8

# <div style="padding:18px;border-width:5px;border-radius:10px;background:linear-gradient(to right, darkblue, red)"> 
# <h1 style="text-align:center; font-size:40px; font-weight:bold; color:white">Mapping Haiti’s Food Prices: A Geo-Spatial Analysis</h1>
# </div>

# ### Installing required libraries

# In[1]:


get_ipython().run_line_magic('pip', 'install folium')


# In[2]:


import folium
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon


# In[3]:


get_ipython().run_line_magic('pip', 'install pandas numpy seaborn matplotlib scikit-learn')


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# Loading the csv file
haiti_df = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Haiti/cleaned_food_prices.csv")

# Show top 5 rows of dataset
print(haiti_df.head())


# In[6]:


haiti_df.dtypes


# In[7]:


# converting features properly
haiti_df['date'] = pd.to_datetime(haiti_df['date'])

cat_feat = ['commodity', 'Department', 'City', 'market', 'food_type', 'unit']
haiti_df[cat_feat] = haiti_df[cat_feat].astype('category')


# In[8]:


haiti_df.dtypes


# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# #  Data Wrangling for the Map

# In[9]:


# Creating the median price per department
haiti_df2 = haiti_df[['usdprice','commodity','Department','latitude','longitude']]
median_prices = haiti_df2.groupby('Department')['usdprice'].median().reset_index()

# Sorting the median prices
median_prices.sort_values(by='usdprice', ascending=False)


# In[10]:


# Creating coordinates for the 9 departments in the dataset
dept_coords = {
    'Artibonite': [19.45, -72.683333],
    'Centre':[19.15, -72.016667] ,
    "Grande'Anse":[18.616667, -74.083333] ,
    'North': [19.757778, -72.204167],
    'North-East':[19.55, -71.733333] ,
    'North-West':[19.939051, -72.8319] ,
    'South': [18.2, -73.75],
    'South-East': [18.234167, -72.534722],
    'West': [18.539167, -72.335]
}


# In[11]:


# Converting the coordinates to a dataframe
coord_df = pd.DataFrame([
    {'Department': k, 'latitude': v[0], 'longitude': v[1]} 
    for k, v in dept_coords.items()
])


# In[12]:


# Merging the median prices and the coordinates
map_df = pd.merge(median_prices, coord_df, on='Department')
map_df


# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# ## Visualizing the Median Prices per Department

# In[20]:


# Generating the map
haiti_map = folium.Map()

# Adding markers
for _,row in map_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['usdprice'] *5,
        color='crimson',
        fill=True,
        fill_color='orange',
        fill_opacity=0.7,
        popup=folium.Popup(f"{row['Department']}: ${row['usdprice']:.2f}", max_width=250)
    ).add_to(haiti_map)
    
haiti_map


# The display is not what I was hoping for. I wanted it to show Haiti with the markers directly without having to zoom in manually. Here's the solution to this...

# In[27]:


# Generic map
haiti_map = folium.Map()

#Adding the markers
for _, row in median_prices.iterrows():
    all_coord = dept_coords.get(row['Department'])
    if all_coord:
        folium.CircleMarker(
            location=all_coord,
            radius=row['usdprice'] * 10,
            color='white',
            fill= True,
            fill_color='blue',
            fill_opacity=0.6,
            popup= folium.Popup(f"{row['Department']}: ${row['usdprice']:.2f}", max_width= 350)
            
        ).add_to(haiti_map)

# Adjusting the map view to fit all markers with fit_bounds
haiti_map.fit_bounds(list(dept_coords.values()))
haiti_map


# Much better!

# ### Comments:

# The standout prices of 2.85 dollars in Artibonite and 2.60 dollars in North-West on signal unique supply chain challenges or market inefficiencies. Further investigation is needed for and **commodity concentration** and the **distribution networks (imported vs local)**.

# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# # Commodity Concentration

# We'll investigate for the first two department with highest median prices.

# ### For Artibonite

# In[15]:


# Filtering Artibonite again
artibonite_df = haiti_df[haiti_df['Department'] == 'Artibonite']

# Total number of rows for Artibonite
total_count = len(artibonite_df)

# Number of rows for the top 2 commodities
top_2_commodities = artibonite_df['commodity'].value_counts().head(2).index.tolist()
top_2_count = artibonite_df[artibonite_df['commodity'].isin(top_2_commodities)].shape[0]

# Calculating proportion
weight = top_2_count / total_count

print(f"Top 2 commodities account for {weight:.2%} of North West listings.")


# ### For North-West

# In[16]:


# Filtering North-West
north_west_df = haiti_df[haiti_df['Department'] == 'North-West']

# Total number of rows for North-West
total_count2 = len(north_west_df)

# Number of rows for just the top 2 commodities
top_2_commodities2 = north_west_df['commodity'].value_counts().head(2).index.tolist()
top_2_count2 = north_west_df[north_west_df['commodity'].isin(top_2_commodities2)].shape[0]

# Calculating proportion
weight = top_2_count2 / total_count2

print(f"Top 2 commodities account for {weight:.2%} of North-West listings.")


# #### Let's represent the weights visually.

# In[31]:


# Weight dictionary
weights = {'Artibonite': 0.3223, 'North-West': 0.3636}
# Loop through weights to create CircleMarkers
for dept, weight in weights.items():
    coords = dept_coords[dept]
    folium.CircleMarker(
        location=coords,
        radius=weight * 50,  
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.9,
        popup=f"{dept}: Top 2 commodities represent {weight:.2%}"
    ).add_to(haiti_map)
haiti_map


# ### Comments:

# 32.23% and 36.36% tell us that market activity in both Artibonite and North-West is concentrated in two commodities, representing approximately one-third of all staple goods consumed. Therefore, the high price levels in these region may reflect not just inflationary pressure, but a narrow set of frequently traded goods.

# ### Implications

# 1. **Market Dependence**: A heavy reliance on two goods suggests a vulnerability to price shocks if supply is disrupted.
# 
# 2. **Reduced Economic Diversity:** If a region’s markets trade in just a couple of goods, it can signal *limited access to alternative food options*.
# 
# 3. **Price Volatility Risk:** When a large portion of transactions centers around two commodities, any disruption *(import delays or inflation)* can cause large price swings, dragging the entire food market’s price median upward.
# 
# 4. **Aid Policy Programs:** Food subsidies in Artibonite and North-West might be more effective if they target those dominant commodities *(Wheat flour and Maaize meal)*.

# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# <div style="border:2px solid black; padding=:15px; background-color:#f4f4f4;">
# <h1 style="text-align:center;">Summary</h1>
# <p style="text-align:center"> <b>Haiti’s food markets show significant disparities. Artibonite and North-West region have the highest median prices and heavy reliance on two commodities. This spatial imbalance signals vulnerability in access, trade diversity, and price stability.</b>.</p>
# <hr style="border:2px red; background-color: white">
# <h3 style="text-align:center;"> Presented by:
# <a href="https://www.linkedin.com/in/carllegros/" target="_blank" style="color:blue;"> Carl Legros </a>
# </h3>
# </div>
# 
