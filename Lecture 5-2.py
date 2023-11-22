#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

prices = [100, 101, 102]
volumes = [200, 210, 220]

data = {
'Price': prices, 'Volume': volumes
}
df = pd.DataFrame(data)


# In[4]:


date_range = pd.date_range(start='2021-01-04', periods=3, freq='D')

opening_prices = [100.0, 102.5, 98.7] 

df = pd.DataFrame(index=date_range, columns=['Opening Price'], data=opening_prices)

print(df)


# In[5]:


df['Volume'] = [50, 49, 53]
df['High'] = [105.0, 104.2, 101.8]
df['Low'] = [98.5, 100.0, 97.2]

print(df)


# In[11]:


import numpy as np

price_array = np.array([100, 101, 102])

volume_array = np.array([200, 210, 220])

df_from_array = pd.DataFrame({
    'price': price_array,
    'volume': volume_array
})
print(df_from_array)


# In[16]:


closing_prices = np.array([100, 105, 90, 78, 123, 145, 99])

df = pd.DataFrame({
    'Closing Price': closing_prices
})

df['Price Change (pct)'] = df['Closing Price'].pct_change() * 100

print(df)


# In[17]:


volume_trades = np.array([45, 52, 33, 29, 89])
average_trade_size = np.array([150, 142, 129, 112, 160])

df = pd.DataFrame({
    'Volume': volume_trades,
    'average trade size': average_trade_size
})

print(df)


# In[23]:


df = pd.DataFrame({'A': [1, 2, 3]})

dates = pd.date_range(start='2021-01-01', periods=3, freq='B')

df['Date'] = dates

df.set_index('Date', inplace=True)

print(df)


# In[27]:


import yfinance as yf

trading_days = pd.date_range(start = '2022-01-01', end='2022-02-01', freq='B')

closing_prices = np.random.uniform(150, 200, size=len(trading_days)).round(2)
trading_volumes = np.random.randint(100000, 500000, size=len(trading_days))
                                    
df_aapl = pd.DataFrame({
    'Date': trading_days, 
    'Close': closing_prices, 
    'Volume': trading_volumes
})   
                                    
monday_filter = df_aapl['Date'].dt.weekday == 0
df_no_mondays = df_aapl.loc[~monday_filter]

                                    
average_volume_no_mondays = df_no_mondays['Volume'].mean()
                                    
print("DataFrame with One Month of AAPL Trading Data:")
print(df_aapl)

print("\nDataFrame without Mondays:")
print(df_no_mondays)

print("\nAverage Trading Volume for Non-Mondays:", average_volume_no_mondays)                                    


# In[29]:


trading_days = pd.date_range(start= '2023-01-01', end=  '2023-01-31', freq='B')

closing_prices = np.random.uniform(150, 200, size=len(trading_days)).round(2)
trading_volumes = np.random.randint(1000000, 5000000, size=len(trading_days))

df = pd.DataFrame({
    'Date': trading_days, 
    'Close': closing_prices, 
    'Volume': trading_volumes
})

df.set_index('Date', inplace=True)

df['Day of the Week'] = df.index.day_name()

print("DataFrame with Day of the Week Column:")
print(df)


# In[33]:


data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Close': [100, 102, 105]} 

df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date']) 

df.set_index('Date', inplace=True)

mean_close = df['Close'].mean()

print(mean_close)


# In[41]:


date_range = pd.date_range(start='2023-01-01', periods=10, freq='B')

closing_prices = np.random.uniform(150, 200, size=len(date_range)).round(2)

stock_df = pd.DataFrame({'Date': date_range, 'Close': closing_prices})

print("Data Frame with Closing Prices", stock_df)

mean_price = stock_df['Close'].mean()
median_price = stock_df['Close'].median()

print("Mean Closing Price:", mean_price)
print("Median Closing Price:", median_price)


# In[43]:


print("Data Frame with Closing Prices", stock_df)

max_closing_price = stock_df['Close'].max()
min_closing_price = stock_df['Close'].min()

day_max_price = stock_df.loc[stock_df['Close'] == max_closing_price, 'Date'].values[0]
day_min_price = stock_df.loc[stock_df['Close'] == min_closing_price, 'Date'].values[0]

print("\nMaximum Closing Price:", max_closing_price)
print("Minimum Closing Price:", min_closing_price)
print("Day with the Highest Closing Price:", day_max_price)
print("Day with the Lowest Closing Price:", day_min_price)


# In[ ]:




