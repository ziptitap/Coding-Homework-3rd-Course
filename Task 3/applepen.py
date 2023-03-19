#!/usr/bin/env python
# coding: utf-8

# ### 1. Состояние склада на каждый день

# In[1]:


import io
import requests
import pandas as pd
import itertools as it


# In[2]:


def read(name):
    return pd.read_csv('out/input/'+name)
    # return pd.read_csv()


# In[3]:


def sell_per_day(df):
    df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
    df['sku_num'] = df['sku_num'].str[6:8]
    sell = df.groupby(pd.Grouper(key='date', freq='D'))['sku_num'].value_counts()
    df1 = sell[sell.index.get_level_values(1) == 'ap']
    df2 = sell[sell.index.get_level_values(1) == 'pe']
    df1 = df1.droplevel(1)
    df2 = df2.droplevel(1)
    return df1, df2  


# In[4]:


from datetime import datetime

def check_dims(ar1, ar2):
    a = ar1.reset_index()
    a['date'] = pd.to_datetime(a.date, format='%Y-%m-%d')
    b = ar2.reset_index()
    
    res = pd.concat([a['date'], b['date']]).drop_duplicates(keep=False)
    
    if res.empty:
        return -1, -1
   
    d1 = datetime.strptime("2006-01-01", '%Y-%m-%d')
    d2 = datetime.strptime(str(res.values)[2:12], '%Y-%m-%d')

    return (d2 - d1).days, d2
    
    


# In[5]:


def left_in_stock(df, name1, name2, product): # name1 = MS-b1-supply.csv product apple or pan
    # df - кол-во проданного товара в кадый день
    idx = df[(df.index.day == 1) | (df.index.day == 15)] # из датасета продаж выбираем продажи в 1 и 15 день
    supply = read(name1) # привезли на склад 
    
    i, d2 = check_dims(supply, idx)
    if i != -1:
        new_row = pd.Series([0], index=[str(d2)[:10]])
        new_row.index = pd.to_datetime(new_row.index, format='%Y-%m-%d')
        new_row.index.name = 'date'
        df = pd.concat([df.iloc[:i], new_row, df.iloc[i:]])
        idx = df[(df.index.day == 1) | (df.index.day == 15)]
        
    df[idx.index] = supply[product] - idx.values # в каждый 1 и 15 день месяца привозят новый товар => нам надо в эти дни 
                        # прибавить привезенный и вычесть то что продали
        
    idx  = df[(df.index.day != 1) & (df.index.day != 15)]
    df[idx.index] *= -1 # умножаем на -1 все дни которые не 1 и не 15, потому что it.accumulate умеет тока складывать
    
    on_stock = read(name2) # в конце месяца у нас есть данные сколько продуктов осталось на складе на самом деле
                              # эти данные отличаются от полученных мной(изза краж) соглавно заданию нам надо в конце кажд месяца
                              # заменить данные посчитанные мной на известные. Что я и буду ща делать
    on_stock['date'] = pd.to_datetime(on_stock.date, format='%Y-%m-%d')
    df[on_stock.date] = on_stock[product]
    
    daily_on_stock = []  # ну а терь считам скока осталось на складе каждый день 
    # day 01.1: supply - sell_1
    # day 01.2: supply - sell_1 - sell_2
    # day 01.3: supply - sell_1 - sell_2 - sell_3
    # ...
    # day 01.15: day 14 + supply - sell_15
    # day 01.31: on_stock[product]
    # day 02.1: on_stock[product] - sell_32
    for i in range(len(on_stock)):
        if i == 0:
            ss = df[(df.index < on_stock.date[i])]
        else:
            ss = df[(df.index >= d) & (df.index < on_stock.date[i])]
            
        daily_on_stock += list(it.accumulate(ss))
        d = on_stock.date[i]
        
    daily_on_stock.append(int(on_stock[product].tail(1).values)) 
    return daily_on_stock


# In[6]:


def check_daily_on_stock(my_ans_apple, my_ans_pen, ans):
    res1 = pd.concat([pd.Series(my_ans_apple), ans['apple']]).drop_duplicates(keep=False)
    res2 = pd.concat([pd.Series(my_ans_pen), ans['pen']]).drop_duplicates(keep=False)
    return res1.empty and res2.empty


# In[7]:


names_sell = ['MS-b1-sell.csv', 
         'MS-b2-sell.csv',
         'MS-m1-sell.csv',
         'MS-m2-sell.csv',
         'MS-s1-sell.csv',
         'MS-s2-sell.csv',
         'MS-s3-sell.csv',
         'MS-s4-sell.csv',
         'MS-s5-sell.csv']

names_supply = ['MS-b1-supply.csv', 
         'MS-b2-supply.csv',
         'MS-m1-supply.csv',
         'MS-m2-supply.csv',
         'MS-s1-supply.csv',
         'MS-s2-supply.csv',
         'MS-s3-supply.csv',
         'MS-s4-supply.csv',
         'MS-s5-supply.csv']

names_inventory = ['MS-b1-inventory.csv', 
         'MS-b2-inventory.csv',
         'MS-m1-inventory.csv',
         'MS-m2-inventory.csv',
         'MS-s1-inventory.csv',
         'MS-s2-inventory.csv',
         'MS-s3-inventory.csv',
         'MS-s4-inventory.csv',
         'MS-s5-inventory.csv']


# In[ ]:


apple_on_stock = []
pen_on_stock = []
for name in zip(names_sell, names_supply, names_inventory):
    sell = read(name[0])
    daily_sales_apple, daily_sales_pen = sell_per_day(sell)
    apple_on_stock.append(left_in_stock(daily_sales_apple, name[1], name[2], 'apple'))
    pen_on_stock.append(left_in_stock(daily_sales_pen, name[1], name[2], 'pen'))


# ### 2. Подсчет сворованного товара за каждый месяц

# Загрузим данные

# In[11]:


markets = ['b1', 'b2', 'm1', 'm2', 's1', 's2', 's3', 's4', 's5']   # id'шники магазинов
data = []

for market in markets:
    inventory = pd.read_csv(f'out/input/MS-{market}-inventory.csv')
    sell = pd.read_csv(f'out/input/MS-{market}-sell.csv')
    supply = pd.read_csv(f'out/input/MS-{market}-supply.csv')
    data.append((inventory, sell, supply))


# Переведем в удобный формат (присвоим индексам таблицы дату)

# In[12]:


dataReserve = data
for market in dataReserve:
    for file in market:
        file['date'] = pd.to_datetime(file.date, format='%Y-%m-%d')
        file.index = file.date
        file.drop('date', axis=1, inplace=True)


# Создадим нужный временной интервал

# In[13]:


daterange = pd.date_range('2006-01','2015-12', freq='MS').strftime("%Y-%m").tolist()


# Подсчет украденных яблок и карандашей для каждого магазина

# In[14]:


rows, result = [], []

for market, name in zip(dataReserve, markets):
    if rows:
        result.append(pd.DataFrame(rows, columns=columns))
    columns = ['date', 'apple', 'pen']
    rows = []
    inventory_apple_at_the_end_of_month = 0
    inventory_pen_at_the_end_of_month = 0
    for month in daterange:
        total_apple_supply_in_month = market[2].loc[month].apple.sum() + inventory_apple_at_the_end_of_month            
        total_pen_supply_in_month = market[2].loc[month].pen.sum() + inventory_pen_at_the_end_of_month
        total_apple_sell_in_month = len(market[1].loc[month][market[1].loc[month]['sku_num'].str.match(f'MS-{name}-a')])
        total_pen_sell_in_month = len(market[1].loc[month][market[1].loc[month]['sku_num'].str.match(f'MS-{name}-p')])
        inventory_apple_at_the_end_of_month = market[0].loc[month].apple.sum()
        inventory_pen_at_the_end_of_month = market[0].loc[month].pen.sum()        
        stolen_apples = total_apple_supply_in_month - total_apple_sell_in_month - inventory_apple_at_the_end_of_month
        stolen_pens = total_pen_supply_in_month - total_pen_sell_in_month - inventory_pen_at_the_end_of_month
        row = [month, stolen_apples, stolen_pens]
        rows.append(row)
        
result.append(pd.DataFrame(rows, columns=columns))


# In[15]:


result[8] # stealed goods from s5 market


# Сохраним таблицы сворованных товаров в csv

# In[16]:


for df, name in zip(result, markets):
    df.to_csv(f'MS-{name}-steal.csv', index=False)

