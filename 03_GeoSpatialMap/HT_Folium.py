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
#     Mapping Haiti's Food Prices: A Geo-Spatial Analysis
# 
#   </h1>
# </div>

# <br></br>

# In[38]:


# Install folium
get_ipython().run_line_magic('pip', 'install folium')


# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import folium
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon
import json
import requests


# In[40]:


# Load dataset
file_id = "1VyLHX20ofimGB7jhFhhw06MBu9Ajod4o"
url = f"https://drive.google.com/uc?download&id={file_id}"
haiti_df = pd.read_csv(url)
print(haiti_df.head())


# In[41]:


# Data types
haiti_df.dtypes


# In[42]:


# Convert date feature
haiti_df['date'] = pd.to_datetime(haiti_df['date'])


# In[43]:


# Data types check
haiti_df.dtypes


# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# #  **Data Preprocessing**

# In[44]:


# Creating the median price per department
haiti_df2 = haiti_df[['usdprice','commodity','Department','latitude','longitude']]
median_prices = haiti_df2.groupby('Department')['usdprice'].median().reset_index()

# Sorting the median prices
median_prices = median_prices.sort_values(by='usdprice', ascending=False)
print(median_prices)


# In[45]:


# Create coordinates for the 9 departments in the dataset
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


# In[46]:


# Convert the coordinates to a dataframe
coord_df = pd.DataFrame([
    {'Department': k, 'latitude': v[0], 'longitude': v[1]}
    for k, v in dept_coords.items()
])


# In[47]:


# Merge median prices and the coordinates
map_df = pd.merge(median_prices, coord_df, on='Department')
map_df


# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# # **Map**

# In[48]:


# Generic map
haiti_map = folium.Map()

# Add the markers
for _, row in median_prices.iterrows():
    all_coord = dept_coords.get(row['Department'])
    if all_coord:
        folium.CircleMarker(
            location=all_coord,
            radius=row['usdprice'] * 10,
            fill= True,
            fill_color='blue',
            fill_opacity=0.6,
            popup= folium.Popup(f"{row['Department']}: ${row['usdprice']:.2f}", max_width= 350)

        ).add_to(haiti_map)
MousePosition().add_to(haiti_map)

# Adjust the map view to fit all markers with fit_bounds
haiti_map.fit_bounds(list(dept_coords.values()))
haiti_map


# ### Comment:

# The standout prices of 2.85 dollars in Artibonite and 2.60 dollars in North-West on signal unique supply chain challenges or market inefficiencies.

# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# # Commodity Concentration

# We'll investigate for the first two department with highest median prices.

# ### For Artibonite

# In[49]:


# Filter Artibonite
artibonite_df = haiti_df[haiti_df['Department'] == 'Artibonite']

# Total number of rows for Artibonite
total_count = len(artibonite_df)

# Number of rows for the top 2 commodities
top_2_commodities = artibonite_df['commodity'].value_counts().head(2).index.tolist()
top_2_count = artibonite_df[artibonite_df['commodity'].isin(top_2_commodities)].shape[0]

# Calculate proportion
weight = top_2_count / total_count

print(f"Top 2 commodities account for {weight:.2%} of North West listings.")


# ### For North-West

# In[50]:


# Filter North-West
north_west_df = haiti_df[haiti_df['Department'] == 'North-West']

# Total number of rows for North-West
total_count2 = len(north_west_df)

# Number of rows for just the top 2 commodities
top_2_commodities2 = north_west_df['commodity'].value_counts().head(2).index.tolist()
top_2_count2 = north_west_df[north_west_df['commodity'].isin(top_2_commodities2)].shape[0]

# Calculate proportion
weight = top_2_count2 / total_count2

print(f"Top 2 commodities account for {weight:.2%} of North-West listings.")


# #### Let's represent the weights visually.

# In[51]:


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
MousePosition().add_to(haiti_map)

haiti_map


# #### Comment:
# 32.23% and 36.36% show that both Artibonite and North‑West rely heavily on just two commodities, making up about one‑third of all staple food listings. **This concentration suggests that higher prices in these regions may come not only from inflation, but also from the fact that a small set of goods dominates both local trade and local consumption.**

# <br></br>

# ## What does this reveal for Haiti's food market dynamics?

# 1. **Market Dependence**: A heavy reliance on two goods suggests a vulnerability to price shocks if supply is disrupted.
# 
# 2. **Reduced Economic Diversity:** If a region's markets trade in just a couple of goods, it can signal limited access to alternative food options.
# 
# 3. **Price Volatility Risk:** When a large portion of transactions centers around two commodities, any disruption *(import delays or inflation)* can cause large price swings, dragging the entire food market's price median upward.
# 
# 4. **Aid Policy Programs:** Food subsidies in Artibonite and North-West might be more effective if they target those dominant commodities *(Wheat flour and Maize meal)*.

# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# # **Choropleth**

# In[52]:


# Load json file
file_id = "1REqI1qWXQoTdNtbiU7P22W8Vn5mvXnS6"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

response = requests.get(url)
world_geo = response.json()

print("Loaded GeoJSON with", len(world_geo), "top-level keys")


# In[53]:


# Making sure the Department names are the same
geo_names = [feature['properties']['name'] for feature in world_geo['features']]
df_names = median_prices['Department'].unique()

print("Names in JSON but not in DataFrame:", set(geo_names) - set(df_names))
print("Names in DataFrame but not in JSON:", set(df_names) - set(geo_names))


# In[54]:


# Department name mapping (in french for the GeoJSON file)
name_mapping = {
    'North-West': 'Nord-Ouest',
    'North-East': 'Nord-Est',
    'South': 'Sud',
    'South-East': 'Sud-Est',
    'West': 'Ouest',
    'Artibonite': "L'Artibonite",
    'North': 'Nord',
    "Grande'Anse": "Grand'Anse"}


# In[55]:


# Copy original median prices dataframe
median_prices_cleaned = median_prices.copy()

# Apply the name translation in the copied dataset median_price_cleaned
median_prices_cleaned['Department'] = median_prices_cleaned['Department'].replace(name_mapping)


# In[56]:


# Create the base map
haiti_map = folium.Map(location=[19.0, -72.7], zoom_start=9)

# Add Choropleth
folium.Choropleth(
    geo_data=world_geo,
    data=median_prices_cleaned,
    columns=['Department', 'usdprice'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Median Prices per Department',
    highlight=True
).add_to(haiti_map)

# Display the map
haiti_map


# The black color Department is likely a department that exists in the GeoJSON but doesn't have a matching entry in the median_prices_cleaned dataset. **Folium renders unmatched figure in black when no value is found**. Here's a way to fix this...

# In[57]:


# Department names in json file
geo_names = [feature['properties']['name'] for feature in world_geo['features']]

# Department names for original dataset
df_names = median_prices_cleaned['Department'].tolist()

# Find unmatched Department
unmatched = set(geo_names) - set(df_names)
print("Departments in GeoJSON but not in DataFrame:", unmatched)


# **Nippes is the unmatched department.**

# Let's assign an average medan price value for `Nippes` with the overall median of all department.

# In[58]:


# Overall median prices for availble departments
overall_median = median_prices_cleaned['usdprice'].median()

# Assign overall median prices for Nippes department
median_prices_cleaned.loc[len(median_prices_cleaned)] = ['Nippes', overall_median]
print('The overall median price for Nippes is:','$',overall_median)


# In[59]:


# Create the base map
haiti_map = folium.Map(location=[19.0, -72.7], zoom_start=9)

# Add Choropleth
folium.Choropleth(
    geo_data=world_geo,
    data=median_prices_cleaned,
    columns=['Department', 'usdprice'],
    key_on='feature.properties.name',
    fill_color='RdBu_r',
    fill_opacity=0.45,
    line_opacity=0.5,
    legend_name='Median Prices per Department',
    highlight=True
).add_to(haiti_map)
MousePosition().add_to(haiti_map)

# Highlight the top 2 high price zones
locs = [19.50853,-73.23007 ] #location to to include the DivIcon text
folium.Marker(
        location=locs,
        icon=DivIcon(html="<div style='font-size:29px; font-weight:bold;color:black;'> High Price Zone</div>")
        ).add_to(haiti_map)
# Display the map
haiti_map


# <hr style="background:linear-gradient(to right,blue,red)"> </hr>

# #### Comment:
# It is noticeable that Haiti's food markets aren't equal across regions. **Artibonite and North‑West have the highest median prices and rely heavily on just two main commodities.** This means some areas have fewer food options, depend on a small set of products, and deal with more unstable prices.

# <br></br>

# ### Presented by:
# <a href="https://www.linkedin.com/in/carllegros/" target="_blank" style="color:blue;"> Carl Legros </a>
