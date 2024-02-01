# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:08:07 2024

@author: juan_
"""
#%%
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#%%
for dirname, _, filenames in os.walk('inputs/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#%%

current_path = os.getcwd()
print("Current Working Directory: ",current_path)

DATA_PATH= current_path + "\\inputs\\"

FILE_FACT = 'fact_table.csv'
FILE_TRANS = 'trans_dim.csv'
FILE_CUSTOMER = 'customer_dim.csv'
FILE_TIME = 'time_dim.csv'
FILE_ITEM = 'item_dim.csv' 
FILE_STORE = 'store_dim.csv'

# Ejemplo
print(f"Ruta del archivo: {FILE_CUSTOMER}")
print(os.path.join(DATA_PATH, FILE_CUSTOMER))

#%%
# Leemos con pandas fact file
fact = pd.read_csv(os.path.join(DATA_PATH, FILE_FACT),encoding='ISO-8859-1' )

fact.rename(
    columns={'quantity': 'fact_quantity', 'unit': 'fact_unit',
             'unit_price':'fact_unit_price','total_price':'fact_total_price'},
    inplace=True
    )

fact.head()

fact.describe()

fact.info()

#%%
# Leemos con pandas trans file
trans = pd.read_csv(os.path.join(DATA_PATH, FILE_TRANS),encoding='ISO-8859-1' )

trans.head()

trans.describe()

trans.info()

#%%
# Leemos con pandas customer file
customer = pd.read_csv(os.path.join(DATA_PATH, FILE_CUSTOMER),encoding='ISO-8859-1' )

customer.rename(
    columns={'name': 'customer_name', 'contact_no': 'customer_contact','nid':'customer_nid' },
    inplace=True
    )

customer.head()

customer.describe()

customer.info()

#%%
# Leemos con pandas items file
time_dim = pd.read_csv(os.path.join(DATA_PATH, FILE_TIME),encoding='ISO-8859-1' )

## convierte date
time_dim['date'] = pd.to_datetime(time_dim['date'] , errors='coerce' )
time_dim['year_month'] = time_dim['month'].astype(str).str.zfill(2) +"-" + time_dim['year'].astype(str) 

time_dim.head()

time_dim.describe()

time_dim.info()

#%%
# Leemos con pandas items file
items = pd.read_csv(os.path.join(DATA_PATH, FILE_ITEM),encoding='ISO-8859-1' )

items.rename(
    columns={'desc': 'item_desc', 'unit_price': 'item_unit_price',
             'man_country':'item_man_country','supplier':'item_supplier',
             'unit':'item_unit'},
    inplace=True
    )

items.head()

items.describe()

items.info()

items.dtypes
#%%
# Leemos con pandas store file
store = pd.read_csv(os.path.join(DATA_PATH, FILE_STORE),encoding='ISO-8859-1' )

store.rename(
    columns={'division': 'store_region', 'district': 'store_district',
             'upazila':'store_sub_district'},
    inplace=True
    )

store.head()

store.describe()

store.info()

#%%

data_consolided = fact.merge(
    trans,
    on=['payment_key'],
    how='left')

data_consolided.info()
#%%
data_consolided = data_consolided.merge(
    customer,
    on=['coustomer_key'],
    how='left')

data_consolided.info()

#%%
data_consolided = data_consolided.merge(
    time_dim,
    on=['time_key'],
    how='left')

data_consolided.info()

#%%
data_consolided = data_consolided.merge(
    items,
    on=['item_key'],
    how='left')

data_consolided.info()

#%%
data_consolided = data_consolided.merge(
    store,
    on=['store_key'],
    how='left')

data_consolided.info()

#%%
# Save results
data_consolided.to_csv(
    # nombre del archivo
    'results/ecommerce_data_processed.csv', 
    # flag para no escribir el indice del dataframe al csv
    index=False
    )

data_consolided.describe()
data_consolided.info()
data_consolided.head()
