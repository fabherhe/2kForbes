import pandas as pd

# Load the data
data_path = '/work/forbes_top_2000_world_largest_public_companies.csv'
df = pd.read_csv(data_path)

# Remove the '$', ',', ' B', and ' M', and replace 'B' with 'e9' and 'M' with 'e6', then evaluate the expression
for column in ['sales', 'profits', 'assets', 'market_value']:
    df[column] = df[column].str.replace('$', '').str.replace(',', '').str.replace(' B', 'e9').str.replace(' M', 'e6')
    df[column] = df[column].apply(lambda x: pd.eval(x) if pd.notnull(x) else np.nan)

df.rename(columns={'contry/territory': 'country'}, inplace=True)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Sales histogram
axs[0, 0].hist(df['sales'], bins=50, color='blue', alpha=0.7)
axs[0, 0].set_title('Sales')

# Profits histogram
axs[0, 1].hist(df['profits'], bins=50, color='green', alpha=0.7)
axs[0, 1].set_title('Profits')

# Assets histogram
axs[1, 0].hist(df['assets'], bins=50, color='red', alpha=0.7)
axs[1, 0].set_title('Assets')

# Market Value histogram
axs[1, 1].hist(df['market_value'], bins=50, color='purple', alpha=0.7)
axs[1, 1].set_title('Market Value')

# Set a layout and show the plot
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot of Sales vs Profits for all companies
axs[0, 0].scatter(df['sales'], df['profits'], alpha=0.5)
axs[0, 0].set_xlabel('Sales')
axs[0, 0].set_ylabel('Profits')
axs[0, 0].set_title('All companies')

# Scatter plot of Sales vs Profits for top 500 companies
top_500 = df.iloc[:500]
axs[0, 1].scatter(top_500['sales'], top_500['profits'], alpha=0.5)
axs[0, 1].set_xlabel('Sales')
axs[0, 1].set_ylabel('Profits')
axs[0, 1].set_title('Top 500 companies')

# Scatter plot of Sales vs Profits for top 100 companies
top_100 = df.iloc[:100]
axs[1, 0].scatter(top_100['sales'], top_100['profits'], alpha=0.5)
axs[1, 0].set_xlabel('Sales')
axs[1, 0].set_ylabel('Profits')
axs[1, 0].set_title('Top 100 companies')

# Scatter plot of Sales vs Profits for top 10 companies
top_10 = df.iloc[:10]
axs[1, 1].scatter(top_10['sales'], top_10['profits'], alpha=0.5)
axs[1, 1].set_xlabel('Sales')
axs[1, 1].set_ylabel('Profits')
axs[1, 1].set_title('Top 10 companies')

plt.tight_layout()
plt.show()

# Identify the company with sales > 5e11
high_sales_company = df[df['sales'] > 5e11]
high_sales_company

# Convert the 'rank' column to numeric by removing the '#' and converting to int
df['rank'] = df['rank'].str.replace('#', '').astype(int)

# Calculate correlation between rank and the other numeric columns
correlation = df[['rank', 'sales', 'profits', 'assets', 'market_value']].corr()

correlation

from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
X = df[['sales', 'profits', 'assets', 'market_value']]
y = df['rank']
model.fit(X, y)

# Get the coefficients
coefficients = model.coef_

coefficients

# Create a copy of the top 10 companies
top_10 = df.iloc[:10].copy()

# Predict the ranking for the top 10 companies
top_10.loc[:, 'predicted_rank'] = model.intercept_ + (-5.80e-09 * top_10['sales']) + (-1.28e-08 * top_10['profits']) + (-2.93e-10 * top_10['assets']) + (-3.66e-10 * top_10['market_value'])

# Sort the top 10 companies by the predicted rank
top_10_sorted = top_10.sort_values('predicted_rank')

top_10_sorted[['company', 'predicted_rank']]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Define a list of dataframes
dfs = [df, df.iloc[:500], df.iloc[:100], df.iloc[:10]]
titles = ['All companies', 'Top 500 companies', 'Top 100 companies', 'Top 10 companies']

# Calculate the total for all companies
total_sales_all = df['sales'].sum()
total_profits_all = df['profits'].sum()
total_assets_all = df['assets'].sum()
total_market_value_all = df['market_value'].sum()

for ax, df_subset, title in zip(axs.flatten(), dfs, titles):
    # Calculate totals
    total_sales = df_subset['sales'].sum()
    total_profits = df_subset['profits'].sum()
    total_assets = df_subset['assets'].sum()
    total_market_value = df_subset['market_value'].sum()
    
    # Calculate the percentages
    sales_percent = total_sales / total_sales_all * 100
    profits_percent = total_profits / total_profits_all * 100
    assets_percent = total_assets / total_assets_all * 100
    market_value_percent = total_market_value / total_market_value_all * 100
    
    # Put totals and percentages in dictionaries
    totals = {'Sales': total_sales, 'Profits': total_profits, 'Assets': total_assets, 'Market Value': total_market_value}
    percentages = {'Sales': sales_percent, 'Profits': profits_percent, 'Assets': assets_percent, 'Market Value': market_value_percent}
    
    # Sort the totals in descending order
    totals_sorted = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    # Create a bar plot of the totals
    bars = ax.bar(totals_sorted.keys(), totals_sorted.values(), color=['blue', 'green', 'red', 'purple'])
    ax.set_title(title)
    ax.set_ylabel('Total')

    # Add data labels
    for bar, metric in zip(bars, totals_sorted.keys()):
        yval = bar.get_height()
        label = f"${yval / 1e12:.2f}T ({percentages[metric]:.1f}%)"
        ax.text(bar.get_x() + bar.get_width()/2, yval, label, ha='center', va='bottom')

plt.tight_layout()
plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Patch

# Create a dictionary to map the country names in your data to the names in the Geopandas data
country_name_mapping = {
    'United States': 'United States of America',
}

# Replace the country names in your data
df['country'] = df['country'].replace(country_name_mapping)

# Load the world geometry data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Count the number of companies in each country
country_counts = df['country'].value_counts()

# Merge the world data with the company counts
world = world.set_index('name').join(country_counts)

# Calculate the maximum count of companies
max_count = country_counts.max()

# Calculate the normalized count for each country
world['normalized_count'] = world['country'] / max_count

# Create a colormap with red, yellow, and green colors
cmap = plt.cm.get_cmap('RdYlGn')

# Plot the world data with a color map based on the normalized count of companies
fig, ax = plt.subplots(1, 1)
world.plot(column='normalized_count', ax=ax, legend=False, cmap=cmap, edgecolor=None)

# Create a scalar mappable object for colorbar
norm = plt.Normalize(vmin=0, vmax=max_count)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set an empty array to avoid error

# Add colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_ticks(np.linspace(0, max_count, 5))
cbar.set_ticklabels(['{:,.0f}'.format(x) for x in np.linspace(0, max_count, 5)])

# Create custom legend patches
legend_patches = [Patch(color=cmap(0), label='0'),
                  Patch(color=cmap(0.25), label='{}K'.format(int(max_count * 0.25 / 1000))),
                  Patch(color=cmap(0.5), label='{}K'.format(int(max_count * 0.5 / 1000))),
                  Patch(color=cmap(0.75), label='{}K'.format(int(max_count * 0.75 / 1000))),
                  Patch(color=cmap(1), label='{}K+'.format(int(max_count / 1000)))]


# Set the title
ax.set_title('Count of Companies by Country')

# Remove the axis labels
ax.set_xticks([])
ax.set_yticks([])

plt.show()