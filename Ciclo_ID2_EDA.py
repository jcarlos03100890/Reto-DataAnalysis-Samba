# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:21:05 2024

@author: juan_
"Exploratory Data Analysis (EDA)"
"""


# %%

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# %%

for dirname, _, filenames in os.walk('results/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%

current_path = os.getcwd()
print("Current Working Directory: ",current_path)

DATA_PATH= current_path + "\\results\\"
FILE_CONSOLIDATED_DATA = 'ecommerce_data_processed.csv'

# Ejemplo
print(f"Ruta del archivo: {FILE_CONSOLIDATED_DATA}")
print(os.path.join(DATA_PATH, FILE_CONSOLIDATED_DATA))


# %%

# Lista de columnas a interpretar como fecha
columns_dates=[
    'date'
    ]

# Lectura del archivo csv
sales_data = pd.read_csv(
    os.path.join(DATA_PATH, FILE_CONSOLIDATED_DATA),
    parse_dates=columns_dates
    )

# %%

sales_data.info()
sales_data.columns

# %%
'''Transaction Type Distribution'''


# %%

print("Unique transaction types:", sales_data['trans_type'].nunique())
print(sales_data['trans_type'].value_counts())
print("Missing bank names:", sales_data['bank_name'].isnull().sum())
print("Unique banks:", sales_data['bank_name'].nunique())
print(sales_data['bank_name'].value_counts())


# %%

#Numero de Transacciones por banco

trans_type_by_bank = sales_data.groupby(
    'bank_name').agg(
        {'payment_key':'count',
         'fact_total_price':'sum'}).reset_index(
             ).rename(columns={'payment_key':'number_of_purchases','fact_total_price': 'total_spent'})

trans_type_by_bank.head(10)
trans_type_by_bank.sample()
trans_type_by_bank.info()

trans_type_by_bank.to_csv(
    # nombre del archivo
    'results/EDA/transations_by_bank.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%
#Transaction by bank
fig = px.bar(trans_type_by_bank.sort_values(by='total_spent', ascending=False).head(15), y=['total_spent','number_of_purchases'], x='bank_name',
              title='Top Bank revenue' ,orientation = "v",  barmode = 'group', labels={'total_spent':'Revenue','number_of_purchases':'Num Sales'})

fig.update_layout(xaxis_title='Bank', yaxis_title='Revenue')
fig.write_html('results/HTML/transations_by_bank.html')
#fig.show()


# %%

#Numero de Transacciones por tipo de transaccion.

trans_type_agg = sales_data.groupby(
    'trans_type').agg(
        {'payment_key':'count',
         'fact_total_price':'sum'}).reset_index(
             ).rename(columns={'payment_key':'number_of_purchases','fact_total_price': 'total_spent'})

trans_type_agg.head()
trans_type_agg.sample()
trans_type_agg.info()

trans_type_agg.to_csv(
    # nombre del archivo
    'results/EDA/transations_distributrion.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )


# %%

fig = plt.figure(figsize=(6,6), dpi=100)
sns.barplot(x='trans_type', y='number_of_purchases', data=trans_type_agg, palette='magma')
plt.title('Transaction Type Distribution')

plt.title('Transaction Type Distribution')
plt.xlabel('Number of purchases', fontsize=16, labelpad=20, color='maroon')
plt.ylabel('Transaction Type', fontsize=16, labelpad=20, color='maroon')

plt.savefig('results/EDA/TransactionTypeDistribution.png')
#plt.show()


# %%

fig = px.bar(trans_type_agg, x='trans_type', y='number_of_purchases', 
             title="Transaction Type", hover_data=['trans_type', 'number_of_purchases','total_spent'], color='trans_type',
             labels={'total_spent':'Total Spent','number_of_purchases':'Num Sales','trans_type':'Transaction type'}, height=600,text_auto=True)
fig.write_html("results/HTML/TransactionTypeDistribution.html")
#fig.show()


# %%

#Pie Chart Transacction type
trans_type_agg.index=trans_type_agg['trans_type']
fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
myexplode = [0.2, 0, 0]
mycolors = ["cyan", "yellow", "green"]
#labels = trans_type_agg['trans_type'].list()
trans_type_agg.plot.pie(y='number_of_purchases', ax=ax, autopct='%1.2f%%', startangle=0, fontsize=12, 
                        label="Transaction Type", explode = myexplode, shadow = True,
                        colors = mycolors)
plt.legend()
plt.title('Transaction Type Distribution')
plt.savefig('results/EDA/Pie-TransactionTypeDistribution.png')

#plt.show()


# %%

#Pie chart using px

fig = px.pie(trans_type_agg, names='trans_type', values='number_of_purchases', color='number_of_purchases', 
             title="Transaction Type",labels={'trans_type':'Transaction type','number_of_purchases':'Purchases'}, height=600)
fig.update_traces(textposition='inside', textinfo='percent')
fig.write_html("results/HTML/Pie-TransactionTypeDistribution.html")
#fig.show()


# %%

#Pie Chart Transacction type
trans_type_agg.index=trans_type_agg['trans_type']
fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
myexplode = [0.2, 0, 0]
mycolors = ["gray", "yellow", "green"]
#labels = trans_type_agg['trans_type'].list()
trans_type_agg.plot.pie(y='total_spent', ax=ax, autopct='%1.2f%%', startangle=0, fontsize=12, 
                        label="Transaction Type", explode = myexplode, shadow = True,
                        colors = mycolors)
plt.legend()
plt.title('Transaction Type Distribution by Revenue')
plt.savefig('results/EDA/Pie-TransactionTypeDistributionRevenue.png')

#plt.show()


# %%

#Pie chart using px

fig = px.pie(trans_type_agg, names='trans_type', values='total_spent', color='total_spent', 
             title="Transaction Type",labels={'trans_type':'Transaction type','total_spent':'Revenue'}, height=600)
fig.update_traces(textposition='inside', textinfo='percent')
fig.write_html("results/HTML/Pie-TransactionTypeDistributionRevenue.html")
#fig.show()

# %%

'''# Customer Analysis'''

# %%
print("\nUnique names:", sales_data['customer_name'].nunique())
print("Missing names:", sales_data['customer_name'].isnull().sum())
print(sales_data['customer_name'].value_counts())

customer_analysis = sales_data.groupby(
    ['customer_name']).agg(
        # conteo de customers
        {'customer_contact': 'count',
        # suma de los precios de los artículos
        'fact_total_price': 'sum'}
                      ).reset_index().rename(
                          columns={
                              'customer_contact':'number_of_purchases',
                              'fact_total_price': 'total_spent'})
    
customer_analysis.head(10)
customer_analysis.sample()
customer_analysis.info()


customer_analysis.to_csv(
    # nombre del archivo
    'results/EDA/customer_analysis.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%

## Top 10 Customers

print(customer_analysis.sort_values(by='total_spent', ascending=False).head(10))
plt.figure(figsize=(14, 9))
sns.barplot(x='total_spent', y='customer_name', 
            data=customer_analysis.sort_values(by='total_spent', ascending=False).head(10), 
            palette='inferno')

# Enhance the plot's aesthetics
plt.title('Top 10 Customers', fontsize=18, fontweight='bold', color='maroon')
plt.xlabel('Total Spent', fontsize=16, labelpad=20, color='maroon')
plt.ylabel('Customer', fontsize=16, labelpad=20, color='maroon')
plt.xticks(fontsize=14, color='maroon')
plt.yticks(fontsize=14, color='maroon')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()  
plt.savefig('results/EDA/Top10Customers.png')
#plt.show()

# %%

customer_analysis.sort_values(by='total_spent', ascending=False).head(10)
fig = px.bar(customer_analysis.sort_values(by='total_spent', ascending=False).head(20), x='total_spent', y='customer_name',  orientation='h',
             title="Top 10 Customers", hover_data=['customer_name', 'number_of_purchases','total_spent'], color='total_spent',
             labels={'total_spent':'Total Spent','number_of_purchases':'Num Sales','customer_name':'Name'}, height=600,text_auto=True)
fig.write_html("results/HTML/Top10Customers.html")
#fig.show()


# %%

'''Top 20 Customers by Total Spent'''

'''Top 20 Customers by Total Spent
 
# The bar chart of the top 20 customers by total spent clearly shows that the top customer, 
# Pooja, has a significantly higher total spending than the others. 
# There is a gradual decline in spending among the remaining top customers. 
# This reinforces the importance of the top spenders to the overall sales and 
# may suggest that personalized marketing strategies or loyalty programs 
# could be effective for maintaining their engagement.'''

# %%

#Distribution of Total Spent by Customers

fig = px.histogram(customer_analysis, x='total_spent',
                   title='Distribution of Total Spent by Customers',
                   nbins=500000,text_auto=True)  # Adjust the number of bins as needed
fig.update_layout(xaxis_title='Total Spent', yaxis_title='Count of Customers')
fig.write_html('results/HTML/DistributionTotalSpentbyCustomers.html')
#fig.show()

# %%
'''Distribution of Total Spent by Customers'''

'''#|Distribution of Total Spent by Customers
#|
#|The histogram shows that most customers spend in the lower monetary range, 
#|with a very steep drop-off as spending amounts increase. This further indicates 
#|that there are only a few high-spending customers, while the majority spend much less.'''


# %%

#Number of Purchases vs Total Spent by Customers

fig = px.scatter(customer_analysis, x='number_of_purchases', y='total_spent', color='total_spent',
                 hover_name='customer_name', title='Number of Purchases vs Total Spent by Customers',size='total_spent')
fig.update_layout(xaxis_title='Number of Purchases', yaxis_title='Total Spent')
fig.write_html('results/HTML/NumberofPurchasesbyCustomers.html')
#fig.show()

# %%

'''Number of Purchases vs Total Spent by Customers'''

'''#The scatter plot suggests a positive correlation between the number of purchases 
#and the total amount spent, which is to be expected. However, there are customers
# with a high number of purchases but relatively lower total spending, indicating 
# they may be buying less expensive items more frequently. Conversely, 
# there are customers with fewer purchases but higher spending, possibly indicating 
# larger transactions or the purchase of higher-priced items.'''


# %%

#Cumulative Distribution of Total Spent by Customers

customer_analysis_sorted = customer_analysis.sort_values('total_spent', ascending=False)
customer_analysis_sorted['cumulative_spent'] = customer_analysis_sorted['total_spent'].cumsum()
customer_analysis_sorted['cumulative_percentage'] = 100 * customer_analysis_sorted['cumulative_spent'] / customer_analysis_sorted['total_spent'].sum()
customer_analysis_sorted = customer_analysis_sorted.sort_values('total_spent', ascending=False).reset_index()
customer_analysis_sorted
customer_analysis_sorted.to_csv(
    # nombre del archivo
    'results/EDA/CumulativeDistributionofTotalSpentbyCustomers.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )


fig = px.line(customer_analysis_sorted, y='cumulative_percentage',
              title='Cumulative Distribution of Total Spent by Customers')
fig.update_layout(xaxis_title='Number of Customers', yaxis_title='Cumulative Percentage of Total Spent')
fig.update_xaxes(range=[0, 1000])  # Adjust the range as needed
fig.write_html('results/HTML/CumulativeDistributionofTotalSpentbyCustomers.html')
#fig.show()

# %%

'''Cumulative Distribution of Total Spent by Customers'''

'''#|This line graph shows that a small number of customers account for a large percentage
#| of the total spending. This kind of distribution is typical of a Pareto principle
#| (or 80/20 rule) where the majority of sales come from a minority of customers. 
#| This indicates that the business might rely on a core group of high-spending customers.'''


# %%

'''Overall Summary'''

'''#|The customer analysis data and the visualizations suggest that the business has a 
#|wide range of customers, but is heavily supported by a small segment of high-value customers. 
#|The company might benefit from strategies aimed at increasing the spend of lower-tier customers 
#|while maintaining the loyalty of the top-tier ones. Understanding the purchasing patterns and 
#|preferences of these key segments could drive targeted marketing campaigns, 
#|personalized promotions, and tailored product recommendations to enhance 
#|customer value across the board.
#|
#|Given the importance of the top spending customers, a customer relationship management (CRM) 
#|strategy could be particularly effective. Additionally, analyzing the data to understand
#| the factors influencing the higher number of transactions and the high-value purchases
#| could provide actionable insights for business growth and customer satisfaction improvement.
'''

# %%

''' # Product Analysis'''

# %%

sales_data.info()
sales_data.columns

print("\nUnique item names:", sales_data['item_name'].nunique())
print(sales_data['item_name'].value_counts())

# Summary of 'desc' (descriptions)
print("\nDescriptions available:", sales_data['item_desc'].nunique())
print("Top descriptions:", sales_data['item_desc'].value_counts().head(10))

# Summary statistics for 'unit_price'
print("\nUnit Price Statistics:")
print(sales_data['item_unit_price'].describe())

# Unique values and value counts for 'man_country' (Manufacturing Country)
print("\nManufacturing Countries:", sales_data['item_man_country'].nunique())
print(sales_data['item_man_country'].value_counts())

# Unique values and value counts for 'supplier'
print("\nSuppliers:", sales_data['item_supplier'].nunique())
print(sales_data['item_supplier'].value_counts())

# Unique values and value counts for 'unit', including missing values
print("\nUnits (with missing values):", sales_data['item_unit'].nunique())
print("Missing units:", sales_data['item_unit'].isnull().sum())
print(sales_data['item_unit'].value_counts())


# %%

#'''How many sales and total incomes by Suppliers and country

suppliers_performance = sales_data.groupby(
    ['item_supplier','item_man_country']).agg(        
        {'item_key':'count', # suma de los precios de los artículos
        'fact_total_price': 'sum'}
                      ).reset_index().rename(columns={
                          'item_key':'number_of_sales',
                          'fact_total_price':'total_revenue'})


suppliers_performance.head()
suppliers_performance.sample()
suppliers_performance.info()

suppliers_performance.to_csv(
    # nombre del archivo
    'results/EDA/suppliers_performance.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%

ax = sns.barplot(x='total_revenue',y='item_supplier', data=suppliers_performance, palette="dark",
                 order=suppliers_performance.sort_values('total_revenue', ascending=False).item_supplier)

# Add axis labels
ax.set(xlabel='Total Revenue', ylabel='Supplier')
plt.title('Revenue by supplier')
plt.savefig('results/EDA/RevenueBySuppliers.png')
#plt.show()        


# %%

fig = px.bar(suppliers_performance.sort_values('total_revenue', ascending=False), x='total_revenue', y='item_supplier', 
             title="Revenue by supplier", hover_data=['total_revenue', 'item_supplier','number_of_sales'], color='item_supplier',
             labels={'total_revenue':'Total Revenue','number_of_sales':'Num Sales','item_supplier':'Supplier'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/RevenueBySuppliers.html")
#fig.show()


# %%

fig = plt.figure(figsize=(6,6), dpi=100)
g = sns.catplot(
    data=suppliers_performance.sort_values('number_of_sales', ascending=False), kind="bar",
    x="number_of_sales", y="item_supplier", 
    errorbar="sd", palette="dark", alpha=.8, height=6, 
)
plt.title('Sales by supplier')
g.despine(left=True)
g.set_axis_labels("Sales", "Manufacturer")
plt.savefig('results/EDA/SalesBySuppliers.png')
#plt.show()      


# %%


fig = px.bar(suppliers_performance.sort_values('number_of_sales', ascending=False), x='number_of_sales', y='item_supplier', 
             title="Salese by supplier", hover_data=['number_of_sales', 'item_supplier','total_revenue'], color='item_supplier',
             labels={'total_revenue':'Total Revenue','number_of_sales':'Num Sales','item_supplier':'Supplier'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/SalesBySuppliers.html")
#fig.show()


# %%

country_sales = sales_data.groupby('item_man_country')['fact_total_price'].sum().reset_index()

# Count plot for 'man_country'
fig, ax = plt.subplots()
sns.barplot(data=country_sales.sort_values('fact_total_price', ascending=False), x='fact_total_price', y='item_man_country', 
            palette="dark",alpha=.8, ax=ax)
# Add axis labels
ax.set(xlabel='Products', ylabel='Manufacturing Countries')

plt.title('Distribution of Manufacturing Countries')
plt.savefig('results/EDA/DistributionManufacturingCountries.png')
#plt.show()

# %%

fig = px.bar(country_sales.sort_values('fact_total_price', ascending=False), x='fact_total_price', y='item_man_country', 
             title="Distribution of Manufacturing Countries", hover_data=['fact_total_price', 'item_man_country'], 
             color='item_man_country',
             labels={'fact_total_price':'Total Revenue','item_man_country':'Country Supplier'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/DistributionManufacturingCountries.html")
#fig.show()


# %%

'''Store Analysis'''


# %%

# Unique values and value counts for 'region'

print("\nRegions: ", sales_data['store_region'].nunique())
print("\nSales by Regions:\n",sales_data['store_region'].value_counts())
print("\Revenue by Regions:\n",sales_data.groupby('store_region')['fact_total_price'].sum().reset_index())

# Unique values and value counts for 'district'
print("\nDistricts: ", sales_data['store_district'].nunique())
print("\nSales by Districts:\n",sales_data['store_district'].value_counts())
print("\Revenue by Districts:\n",sales_data.groupby('store_district')['fact_total_price'].sum().reset_index())

# Unique values and value counts for 'subbdistrict'
print("\nSub-districts: ", sales_data['store_sub_district'].nunique())
print("\nSales by Sub-district:\n",sales_data['store_sub_district'].value_counts())
print("\Revenue by Sub-district:\n",sales_data.groupby('store_sub_district')['fact_total_price'].sum().reset_index())

# Unique values and value counts for 'store'
print("\nStores: ", sales_data['store_key'].nunique())
print("\nSales by Store:\n",sales_data['store_key'].value_counts())
print("\Revenue by Store:\n",sales_data.groupby('store_key')['fact_total_price'].sum().reset_index())

# %%

'''# Store performance by Region, District, Sub-District'''

# %%
#Store Performance by Location

store_performance_by_location= sales_data.groupby(
    ['store_region','store_district']).agg(        
        {'store_key':'count', # suma de los precios de los artículos
        'fact_total_price': 'sum'}
                      ).reset_index().rename(columns={
                          'store_key':'number_of_sales',
                          'fact_total_price':'total_revenue'})


fig = px.treemap(store_performance_by_location, path=['store_region', 'store_district'], values='total_revenue',
                 title='Store Performance by Region and District')
fig.write_html("results/HTML/store_performance_by_location.html")


# %%

#To Revenue-Generating by Region

region_sales = sales_data.groupby('store_region')['fact_total_price'].sum().reset_index()
print(region_sales.sort_values(by='fact_total_price', ascending=False))
region_sales.to_csv(
    # nombre del archivo
    'results/EDA/region_sales.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

plt.figure(figsize=(12, 8))
sns.barplot(x='fact_total_price', y='store_region', 
            data=region_sales.sort_values(by='fact_total_price', ascending=False), palette='coolwarm')

plt.title('Top Revenue-Generating Regions', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Total Revenue', fontsize=14, labelpad=15, color='navy')
plt.ylabel('Region', fontsize=14, labelpad=15, color='navy')
plt.xticks(fontsize=12, color='navy')
plt.yticks(fontsize=12, color='navy')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()  
plt.savefig('results/EDA/TopRevenue-GeneratingRegions.png')
#plt.show()


# %%

fig = px.bar(region_sales.sort_values(by='fact_total_price', ascending=False), x='fact_total_price', y='store_region', 
             title="Top 10 Revenue-Generating Revenue by Region", hover_data=['fact_total_price', 'store_region'], 
             color='store_region',
             labels={'fact_total_price':'Total Revenue','store_region':'Region'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/TopRevenue-GeneratingRegions.html")
#fig.show()


# %%

#Pie Chart Generating Regions type
region_sales.index=region_sales['store_region']
fig = plt.figure(figsize=(10,10), dpi=300)
ax = plt.subplot(111)
region_sales.plot.pie(y='fact_total_price', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=14, 
                        label="Region")
plt.legend()
plt.title('Revenue-Generating by Regions')
plt.savefig('results/EDA/Pie-TopRevenue-GeneratingRegions.png')


# %%

#Pie chart using px

fig = px.pie(region_sales, names='store_region', values='fact_total_price', color='fact_total_price', 
             title="Revenue-Generating by Regions",labels={'store_region':'Region','fact_total_price':'Total Renenue'}, height=600)
fig.update_traces(textposition='inside', textinfo='percent')
fig.write_html("results/HTML/Pie-TopRevenue-GeneratingRegions.html")
#fig.show()


# %%

#Top 10 Revenue-Generating by District

district_sales = sales_data.groupby('store_district')['fact_total_price'].sum().reset_index()
print(district_sales .sort_values(by='fact_total_price', ascending=False).head(10))

district_sales.to_csv(
    # nombre del archivo
    'results/EDA/district_sales.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

plt.figure(figsize=(12, 8))
sns.barplot(x='fact_total_price', y='store_district', 
            data=district_sales.sort_values(by='fact_total_price', ascending=False).head(10), palette='nipy_spectral_r')

plt.title('Top 10 Revenue-Generating by District', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Total Revenue', fontsize=14, labelpad=15, color='navy')
plt.ylabel('District', fontsize=14, labelpad=15, color='navy')
plt.xticks(fontsize=12, color='navy')
plt.yticks(fontsize=12, color='navy')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()  
plt.savefig('results/EDA/Top10Revenue-GeneratingbyDistrict.png')
#plt.show()

# %%

fig = px.bar(district_sales.sort_values(by='fact_total_price', ascending=False).head(10), x='fact_total_price', y='store_district', 
             title="Top 10 Revenue-Generating Revenue by District", hover_data=['fact_total_price', 'store_district'], 
             color='store_district',
             labels={'fact_total_price':'Total Revenue','store_district':'District'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/Top10Revenue-GeneratingbyDistrict.html")
#fig.show()


# %%

''' # Time Analysis'''


#%%

# Summary statistics for 'year'

print("\nYear Statistics, Sales per year:")
print(sales_data['year'].value_counts())


# %%

# Summary statistics for 'year'

sales_by_year = sales_data.groupby(
    ['year']).agg(        
        {'time_key':'count', # suma de los precios de los artículos
        'fact_total_price': 'sum'}
                      ).reset_index().rename(columns={
                          'time_key':'number_of_sales',
                          'fact_total_price':'total_revenue'})
              

# Creating a bar plot for the distribution of years directly from the DataFrame
plt.figure(figsize=(10,6))
sns.barplot(data=sales_by_year, x='year', y='number_of_sales',palette='dark')
plt.title('Sales Distribution by Years')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate the x labels for better readability if necessary

plt.savefig('results/EDA/SalesDistributionofYears.png')
#plt.show()

# %%
fig = px.bar(sales_by_year, x='year', y='number_of_sales', 
             title="Sales Distribution by Years", hover_data=['year', 'number_of_sales'], 
             color='year',
             labels={'year':'Year','number_of_sales':'Total Sales'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/SalesDistributionofYears.html")
#fig.show()

# %%

plt.figure(figsize=(10,6))
sns.barplot(data=sales_by_year, x='year', y='total_revenue',palette='dark')
plt.title('Distribution of Years')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.xticks(rotation=45)  # Rotate the x labels for better readability if necessary

plt.savefig('results/EDA/RevenueDistributionofYears.png')
#plt.show()

# %%

fig = px.bar(sales_by_year, x='year', y='total_revenue', 
             title="Revenue Distribution by Years", hover_data=['year', 'total_revenue'], 
             color='year',
             labels={'year':'Year','total_revenue':'Total Revenue'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/RevenueDistributionofYears.html")
#fig.show()

# %%

'''More analysis'''

# %%

# Histogram for 'total_price'

sns.histplot(sales_data['fact_total_price'], kde=True, bins=30)
plt.title('Distribution of Total Prices')
plt.savefig('results/EDA/DistributionofTotalPrices.png')
#plt.show()

# %%

#Histogram for 'total_price'

fig = px.histogram(sales_data, x='fact_total_price',
                   title='Distribution of Total Prices',
                   nbins=30,text_auto=True, labels={'fact_total_price':'Price','count':'Sales'})  # Adjust the number of bins as needed
fig.update_layout(xaxis_title='Total Price', yaxis_title='Count of Sales')
fig.write_html('results/HTML/DistributionofTotalPrices.html')
#fig.show()

# %%

# Merging the fact_table with the time_dim on 'time_key'

sales_data.columns

monthly_sales = sales_data.groupby(['year', 'month'])['fact_total_price'].sum().reset_index()
monthly_sales['year_month'] = pd.to_datetime(monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str))
monthly_sales.to_csv(
    # nombre del archivo
    'results/EDA/monthly_sales.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%

plt.figure(figsize=(15,7))
sns.lineplot(x='year_month', y='fact_total_price', data=monthly_sales, marker='o')
plt.title('Total Renenue Over Time (Monthly)')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)  # Rotate the x labels for better readability
plt.grid(True)
plt.savefig('results/EDA/TotalRevenueOverTime_Monthly.png')
#plt.show()


# %%

fig = px.line(monthly_sales,x='year_month', y='fact_total_price',
              title='Total Sales Over Time (Monthly)',labels={'year_month':'Month','fact_total_price':'Total Revenue'})
fig.update_layout(xaxis_title='Year-Month', yaxis_title='Total Sales')
fig.write_html('results/HTML/TotalRevenueOverTime_Monthly.html')
#fig.show()


# %%


#Top 10 Revenue-Generating Items

item_sales = sales_data.groupby('item_name')['fact_total_price'].sum().reset_index()

print(item_sales.sort_values(by='fact_total_price', ascending=False).head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='fact_total_price', y='item_name', 
            data=item_sales.sort_values(by='fact_total_price', ascending=False).head(10), palette='viridis')

plt.title('Top 10 Revenue-Generating Products', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Total Revenue', fontsize=14, labelpad=15, color='navy')
plt.ylabel('Product Name', fontsize=14, labelpad=15, color='navy')
plt.xticks(fontsize=12, color='navy')
plt.yticks(fontsize=12, color='navy')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()  
plt.savefig('results/EDA/Top10Revenue-GeneratingItems.png')
#plt.show()

# %%

fig = px.bar(item_sales.sort_values(by='fact_total_price', ascending=False).head(10), x='fact_total_price', y='item_name', 
             title="Top 10 Revenue-Generating Products", hover_data=['fact_total_price', 'item_name'], 
             color='item_name',
             labels={'fact_total_price':'Total Revenue','item_name':'Products'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/Top10Revenue-GeneratingItems.html")
#fig.show()

# %%

#Top 10 Revenue-Generating Descriptions

desc_sales = sales_data.groupby('item_desc')['fact_total_price'].sum().reset_index()


desc_sales.to_csv(
    # nombre del archivo
    'results/EDA/desc_sales.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

print(desc_sales.sort_values(by='fact_total_price', ascending=False).head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='fact_total_price', y='item_desc', 
            data=desc_sales.sort_values(by='fact_total_price', ascending=False).head(10), palette='coolwarm')

plt.title('Top 10 Revenue-Generating Descriptions', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Total Revenue', fontsize=14, labelpad=15, color='navy')
plt.ylabel('Description', fontsize=14, labelpad=15, color='navy')
plt.xticks(fontsize=12, color='navy')
plt.yticks(fontsize=12, color='navy')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()  
plt.savefig('results/EDA/Top10Revenue-GeneratingDescriptions.png')
#plt.show()

# %%

fig = px.bar(desc_sales.sort_values(by='fact_total_price', ascending=False).head(10), x='fact_total_price', y='item_desc', 
             title="Top 10 Revenue-Generating Descriptions", hover_data=['fact_total_price', 'item_desc'], 
             color='item_desc',
             labels={'fact_total_price':'Total Revenue','item_desc':'Description'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/Top10Revenue-GeneratingDescriptions.html")
#fig.show()


# %%

#Total Quantity Sold

item_sales_quantity = sales_data.groupby('item_name')['fact_quantity'].sum().reset_index()

print(item_sales_quantity.sort_values(by='fact_quantity', ascending=False).head(10))

item_sales_quantity.to_csv(
    # nombre del archivo
    'results/EDA/item_sales_quantity.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

plt.figure(figsize=(12, 8))
sns.barplot(x='fact_quantity', y='item_name',
            data=item_sales_quantity.sort_values(by='fact_quantity', ascending=False).head(10), palette='magma')

plt.title('Top 10 Items by Quantity Sold', fontsize=16, fontweight='bold', color='darkred')
plt.xlabel('Total Quantity Sold', fontsize=14, labelpad=15, color='darkred')
plt.ylabel('Item Name', fontsize=14, labelpad=15, color='darkred')
plt.xticks(fontsize=12, color='darkred')
plt.yticks(fontsize=12, color='darkred')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout() 

plt.savefig('results/EDA/Top10ItemsbyQuantitySold.png')

#plt.show()


# %%

fig = px.bar(desc_sales.sort_values(by='fact_total_price', ascending=False).head(10), x='fact_total_price', y='item_desc', 
             title="Top 10 Revenue-Generating Descriptions", hover_data=['fact_total_price', 'item_desc'], 
             color='item_desc',
             labels={'fact_total_price':'Total Revenue','item_desc':'Description'}, 
             height=600,text_auto=True)
fig.write_html("results/HTML/Top10Revenue-GeneratingDescriptions.html")
#fig.show()


# %%

#Top 5 Performing Items in Each Division

division_item_sales = sales_data.groupby(['store_region', 'item_name'])['fact_total_price'].sum().reset_index()

# Sorting the items within each division by total sales in descending order
division_item_sales.sort_values(by=['store_region', 'fact_total_price'], ascending=[True, False], inplace=True)

# Identifying the top 5 performing items in each division
top_5_items_in_division = division_item_sales.groupby('store_region').head(5)

# Displaying the top 5 performing items in each division
print(top_5_items_in_division)

top_5_items_in_division.to_csv(
    # nombre del archivo
    'results/EDA/top_5_items_in_division.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# Creating a bar plot for top 5 items in each division with enhanced aesthetics
plt.figure(figsize=(14, 10))
sns.barplot(x='fact_total_price', y='store_region', hue='item_name', data=top_5_items_in_division, palette='viridis')


plt.title('Top 5 Performing Items in Each Region', fontsize=18, fontweight='bold', color='darkgreen')
plt.xlabel('Total Sales', fontsize=16, labelpad=20, color='darkgreen')
plt.ylabel('Item Name', fontsize=16, labelpad=20, color='darkgreen')
plt.xticks(fontsize=14, color='darkgreen')
plt.yticks(fontsize=14, color='darkgreen')
plt.legend(title='Region', title_fontsize='13', fontsize='12', facecolor='white', edgecolor='black')

plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()  
plt.savefig('results/EDA/Top5PerformingItemsinEachRegion.png')
#plt.show()

# %%

fig = px.bar(top_5_items_in_division, x='fact_total_price', y='store_region', 
             title="Top 5 Performing Items in Each Region", hover_data=['fact_total_price', 'store_region'], 
             color='item_name',
             labels={'fact_total_price':'Total Revenue','store_region':'Region','item_name':'Product'}, 
             height=800,text_auto=True)
fig.update_layout(barmode='group', xaxis={'categoryorder':'category ascending'})
fig.write_html("results/HTML/Top5PerformingItemsinEachRegion.html")
#fig.show()

# %%
g = sns.FacetGrid(top_5_items_in_division, col='store_region', col_wrap=3, height=4, sharex=False)
g.map(sns.barplot, 'fact_total_price', 'item_name', 
      order=top_5_items_in_division['item_name'].unique(), palette='viridis')

g.set_titles('{col_name}')
g.set_axis_labels('Total Sales', 'Item Name')
g.set(ylabel='')

plt.tight_layout()
plt.savefig('results/EDA/Top5PerformingItemsinEachRegionByRegion.png')
#plt.show()

# %%

fig = px.bar(top_5_items_in_division, x='fact_total_price', y='item_name', 
             title="Top 5 Performing Items in Each Region", hover_data=['fact_total_price', 'store_region'], 
             color='item_name', facet_col='store_region', facet_col_wrap=3,
             labels={'fact_total_price':'Total Revenue','store_region':'Region','item_name':'Product'}, 
             height=800)
fig.update_layout(barmode='group', xaxis={'categoryorder':'category ascending'})
fig.write_html("results/HTML/Top5PerformingItemsinEachRegionByRegion.html")
#fig.show()

# %% 
'''# Predictive Analytics#|'''

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# %%

monthly_sales_train = monthly_sales

monthly_sales_train.index = pd.to_datetime(monthly_sales_train['year_month'])

monthly_sales_train.to_csv(
    # nombre del archivo
    'results/Analytics/monthly_sales_train.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )


train_data = monthly_sales_train['fact_total_price'][:-12]  # Hold out the last 12 months for testing
train_data.info()
train_data.to_csv(
    # nombre del archivo
    'results/Analytics/train_data.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

sarima_result = sarima_model.fit()

# Make predictions for the next 12 months
predictions = sarima_result.get_forecast(steps=12)
predicted_means = predictions.predicted_mean
predicted_intervals = predictions.conf_int()

plt.figure(figsize=(15,7))
plt.plot(train_data.index, train_data, label='Observed')
plt.plot(predicted_means.index, predicted_means, color='r', label='Forecast')
plt.fill_between(predicted_intervals.index, predicted_intervals.iloc[:, 0], predicted_intervals.iloc[:, 1], color='pink')
plt.legend()
plt.title('Monthly Sales Forecast')
plt.savefig('results/Analytics/MonthlySalesForecast.png')
#plt.show()



# %%

from sklearn.metrics import mean_absolute_error, mean_squared_error

test_data = monthly_sales['fact_total_price'][-12:]

# Calculate MAE
mae = mean_absolute_error(test_data, predicted_means)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, predicted_means))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Calculate MAPE - Mean Absolute Percentage Error
mape = np.mean(np.abs((test_data - predicted_means) / test_data)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

data = [
    ["Mean Absolute Error (MAE)", mae],
    ["Root Mean Squared Error (RMSE)", rmse],
    ["Mean Absolute Percentage Error (MAPE)", mape]
]

df_error = pd.DataFrame(data, columns=["Erro_type", "Value"])

df_error.to_csv(
    # nombre del archivo
    'results/Analytics/error_type.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )


# %%

predicted_means = predicted_means.reset_index().rename(columns={"index": "year_month"})
train_data = train_data.reset_index()
test_data = test_data.reset_index()

# %%

#fig = px.line(monthly_sales, x='year_month', y='fact_total_price', title='Observed')

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data["year_month"], y=train_data["fact_total_price"], name="Observed", mode="lines"))
fig.add_trace(go.Scatter(x=predicted_means["year_month"], y=predicted_means["predicted_mean"], name="Forecast", 
                         mode="lines", fillcolor='red',))
fig.add_trace(go.Scatter(x=test_data["year_month"], y=test_data["fact_total_price"], name="Observed", 
                         mode="lines", fillcolor='green',))
fig.update_layout(
    title="Monthly Sales Forecast", xaxis_title="Year Month", yaxis_title="Revenue"
)

fig.write_html('results/Analytics/MonthlySalesForecast.html')
#fig.show()


# %%

division_item_sales = sales_data.groupby(['store_region', 'item_name'])['fact_total_price'].sum().reset_index()
division_item_sales.to_csv('results/Analytics/Regions.csv',index=False )
divisions = sales_data['store_region'].unique()
divisions

num_divisions = len(divisions)

fig, axes = plt.subplots(nrows=num_divisions, ncols=1, figsize=(15, 7*num_divisions))

# Loop through each division
for i, division in enumerate(divisions):
    division_data = sales_data[sales_data['store_region'] == division]

    monthly_prices = division_data.groupby(['year', 'month'])['fact_total_price'].sum().reset_index()
    monthly_prices['year_month'] = pd.to_datetime(
        monthly_prices['year'].astype(str) + '-' + monthly_prices['month'].astype(str))

    monthly_prices.index = pd.to_datetime(monthly_sales_train['year_month'])

    train_data = monthly_prices['fact_total_price'][:-12]  # Hold out the last 12 months for testing

    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Fit the model
    sarima_result = sarima_model.fit()

    predictions = sarima_result.get_forecast(steps=12)
    predicted_means = predictions.predicted_mean
    predicted_intervals = predictions.conf_int()

    axes[i].plot(train_data.index, train_data, label='Observed')
    axes[i].plot(predicted_means.index, predicted_means, color='r', label='Forecast')
    axes[i].fill_between(predicted_intervals.index, predicted_intervals.iloc[:, 0], predicted_intervals.iloc[:, 1], color='pink')
    axes[i].legend()
    axes[i].set_title('Monthly Sales Forecast for ' + division)

    predicted_means = predicted_means.reset_index().rename(columns={"index": "year_month"})
    train_data = train_data.reset_index().rename(columns={"index": "year_month"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data["year_month"], y=train_data["fact_total_price"], name="Observed", mode="lines"))
    fig.add_trace(go.Scatter(x=predicted_means["year_month"], y=predicted_means["predicted_mean"], name="Forecast", 
                             mode="lines", fillcolor='red',))
    fig.update_layout(
        title="Monthly Sales Forecast for " + division, xaxis_title="Year Month", yaxis_title="Revenue"
    )

    fig.write_html(f'results/Analytics/MonthlySalesForecastFor{division}.html')

# Adjust layout spacing
plt.tight_layout()

plt.savefig('results/Analytics/MonthlySalesForecastByRegion.png')
# Show all the subplots
#plt.show()

# %%

desc_sales = sales_data.groupby('item_desc')['fact_total_price'].sum().reset_index()

desc_sales.to_csv(
    # nombre del archivo
    'results/Analytics/RevenuebyDescriptions.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )
top_desc_sales = desc_sales.sort_values(by='fact_total_price', ascending=False).head(10)

print(top_desc_sales)

for i in top_desc_sales.item_desc:
    print(i)

top_desc_values = top_desc_sales['item_desc'].tolist()
top_desc_values
subset_df = sales_data[sales_data['item_desc'].isin(top_desc_values)]
subset_df.count()

subset_df.to_csv(
    # nombre del archivo
    'results/Analytics/top_desc_values.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%"

unique_desc = subset_df['item_desc'].unique()
unique_desc
for desc in unique_desc:
    desc_data = subset_df[subset_df['item_desc'] == desc]

    monthly_prices = desc_data.groupby(['year', 'month'])['fact_total_price'].sum().reset_index()

    monthly_prices['year_month'] = pd.to_datetime(
        monthly_prices['year'].astype(str) + '-' + monthly_prices['month'].astype(str))

    monthly_prices.index = pd.to_datetime(monthly_sales_train['year_month'])

    train_data = monthly_prices['fact_total_price'][:-12]  # Hold out the last 12 months for testing

    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    sarima_result = sarima_model.fit()

    predictions = sarima_result.get_forecast(steps=12)
    predicted_means = predictions.predicted_mean
    predicted_intervals = predictions.conf_int()

    plt.figure(figsize=(15,7))
    plt.plot(train_data.index, train_data, label='Observed')
    plt.plot(predicted_means.index, predicted_means, color='r', label='Forecast')
    plt.fill_between(predicted_intervals.index, predicted_intervals.iloc[:, 0], predicted_intervals.iloc[:, 1], color='pink')
    plt.legend()
    plt.title('Monthly Sales Forecast for ' + desc)
    plt.savefig('results/Analytics/MonthlySalesForecastfor' + desc.replace("/", "-") +'.png')
    
    predicted_means = predicted_means.reset_index().rename(columns={"index": "year_month"})
    train_data = train_data.reset_index().rename(columns={"index": "year_month"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data["year_month"], y=train_data["fact_total_price"], name="Observed", mode="lines"))
    fig.add_trace(go.Scatter(x=predicted_means["year_month"], y=predicted_means["predicted_mean"], name="Forecast", 
                             mode="lines", fillcolor='red',))
    fig.update_layout(
        title="Monthly Sales Forecast for " + desc, xaxis_title="Year Month", yaxis_title="Revenue"
    )

    fig.write_html('results/Analytics/MonthlySalesForecastFor'+ desc.replace("/", "-") +'.html')
    
    #plt.show()

# %%

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = sales_data.groupby(['time_key'])['item_desc'].apply(list).values.tolist()

transactions
    

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.to_csv(
    # nombre del archivo
    'results/Analytics/transactions_analytic.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

# %%

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)  # adjust the support as needed
frequent_itemsets

# %%

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)  
rules

# %%

rules.sort_values(by='confidence', ascending=False, inplace=True)
rules

# %%

print(rules[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence', ascending=False))
rules.to_csv(
    # nombre del archivo
    'results/Analytics/rulesAnalytics.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )


# %%

# Scatter plot of Support vs Confidence
plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules)
plt.title('Association Rules Scatter Plot (Support vs Confidence)')
plt.savefig('results/Analytics/AssociationRulesScatterPlot_SupportvsConfidence.png')
#plt.show()


# %%

#Number of Purchases vs Total Spent by Customers
fig = px.scatter(rules, x='support', y='confidence', color='lift',
                 hover_name='support', title='Association Rules Scatter Plot (Support vs Confidence)',size='lift')
fig.update_layout(xaxis_title='Confidence', yaxis_title='Support')
fig.write_html('results/Analytics/AssociationRulesScatterPlot_SupportvsConfidence.html')
#fig.show()

# %%

import networkx as nx

def draw_graph(rules, rules_to_show):
    G1 = nx.DiGraph()

    for i in range(min(len(rules), rules_to_show)):  # Ensure not to exceed the number of rules
        G1.add_nodes_from(["R"+str(i)])
        
        for a in rules.iloc[i]['antecedents']:
            G1.add_nodes_from([a])
            G1.add_edge(a, "R"+str(i), color='orange', weight=2)
            
        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([c])
            G1.add_edge("R"+str(i), c, color='blue', weight=2)

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u, v in edges]
    weights = [G1[u][v]['weight'] for u, v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, node_color='lightblue', edge_color=colors, width=weights, 
            font_size=16, with_labels=True, node_size=3500, arrowsize=20)

    plt.title('Network Graph of Association Rules')
    plt.savefig('results/Analytics/NetworkGraphofAssociationRules.png')
    #plt.show()

   
# Draw the graph for the top 10 rules
draw_graph(rules, 20)

# %%

def recommend_items_with_confidence(user_items, rules, top_n=5):
    """
    Recommend items along with the confidence of the recommendation based on a set of user items.
    
    Parameters:
    user_items: list, the items already chosen or liked by the user.
    rules: DataFrame, the association rules.
    top_n: int, number of items to recommend.
    
    Returns:
    recommendations_with_confidence: list of tuples, each tuple contains an item and its confidence.
    """
    # Filter rules with antecedents in user_items
    applicable_rules = rules[rules['antecedents'].apply(lambda antecedent: antecedent.issubset(set(user_items)))]

    # Sort rules by descending confidence, lift, or other metric
    sorted_rules = applicable_rules.sort_values(by='confidence', ascending=False)

    # Extract consequents and confidence
    recommendations_with_confidence = []
    for _, row in sorted_rules.iterrows():
        for item in row['consequents']:
            if item not in user_items:  # Check if the user already has the item
                recommendations_with_confidence.append((item, row['confidence']))

    # Remove duplicates while preserving order and limit to top_n
    seen = set()
    unique_recommendations = []
    for item, conf in recommendations_with_confidence:
        if item not in seen:
            seen.add(item)
            unique_recommendations.append((item, conf))
        if len(unique_recommendations) == top_n:
            break

    return unique_recommendations

# %%

user_items = ['Coffee Ground', 'Gum - Mints']  
recommended_items_with_confidence = recommend_items_with_confidence(user_items, rules, top_n=20)

print("Recommended items with confidence:")
for item, confidence in recommended_items_with_confidence:
    print(f"Item: {item}, Confidence: {confidence:.2f}")
    
df = pd.DataFrame(recommended_items_with_confidence, columns =['Item', 'Confidence'])

df.to_csv(
    # nombre del archivo
    'results/Analytics/recommended_items_with_confidence.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )
 