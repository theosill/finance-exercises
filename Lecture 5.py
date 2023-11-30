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


# In[11]:


import numpy as np
import pandas as pd
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


# In[14]:


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


# In[1]:





# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Close': [100, 102, 105]} 

df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date']) 

df.set_index('Date', inplace=True)

df['Close'].plot(title='Stock Closing Prices') 
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()


# In[8]:


max_close_day = df['Close'].idxmax()

ax = df['Close'].plot(kind='bar', title='Stock Closing Prices', color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')

ax.patches[df.index.get_loc(max_close_day)].set_facecolor('red')

plt.show()


# In[9]:


import pandas as pd

prices = [100, 101, 102, 103, 105]

price_series = pd.Series(prices)


# In[10]:


dates = pd.date_range('20230101', periods=5) 

price_series = pd.Series(prices, index=dates)


# In[11]:


returns = price_series.pct_change()


# In[12]:


random_stock_prices = pd.Series(np.random.randint(50, 150, 10), name='Stock Prices')

daily_returns = random_stock_prices.pct_change() * 100

daily_returns = daily_returns.dropna()

print("Random Stock Prices:")
print(random_stock_prices)
print("\nDaily Returns (%):")
print(daily_returns)


# In[13]:


closing_prices = pd.Series(np.random.randint(50, 150, 10), name='Closing Prices')


moving_average = closing_prices.rolling(window=3).mean()

plt.figure(figsize=(10, 6))
plt.plot(closing_prices, label='Closing Prices', marker='o')
plt.plot(moving_average, label='3-Day Moving Average', linestyle='--', color='orange', marker='o')
plt.title('Stock Closing Prices and 3-Day Moving Average')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


import pandas as pd

data = {'Symbol': ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'GOOG', 'MSFT'], 
        'Date': ['2023-01-01', '2023-01-01', '2023-01-01',
                 '2023-01-02', '2023-01-02', '2023-01-02'], 
        'Volume': [100, 150, 200, 90, 120, 160]}

trades = pd.DataFrame(data)

grouped = trades.groupby('Symbol')


# In[17]:


total_volume = grouped['Volume'].sum()


# In[19]:


data = {'Symbol': ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'GOOG', 'MSFT'],
        'Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-02'],
        'Volume': [100, 150, 200, 90, 120, 160]}

trades = pd.DataFrame(data)

average_volume_per_day = trades.groupby('Date')['Volume'].mean()

max_volume_day = average_volume_per_day.idxmax()
max_average_volume = average_volume_per_day.max()

print("Trades DataFrame:")
print(trades)
print("\nAverage Trading Volume per Day:")
print(average_volume_per_day)
print("\nDay with the Highest Average Trading Volume:")
print(f"Date: {max_volume_day}, Average Volume: {max_average_volume}")


# In[20]:


data = {'Symbol': ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'GOOG', 'MSFT'],
        'Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-02'],
        'Volume': [100, 150, 200, 90, 120, 160]}
trades = pd.DataFrame(data)

trades['Date'] = pd.to_datetime(trades['Date'])

trades['Price'] = [150, 200, 250, 160, 210, 180]

trades['Total Traded Value'] = trades['Price'] * trades['Volume']
total_traded_value_per_symbol = trades.groupby('Symbol')['Total Traded Value'].sum()

max_traded_symbol = total_traded_value_per_symbol.idxmax()
max_total_traded_value = total_traded_value_per_symbol.max()

print("Extended Trades DataFrame:")
print(trades)
print("\nTotal Traded Value per Stock Symbol:")
print(total_traded_value_per_symbol)
print("\nStock with the Highest Total Traded Value:")
print(f"Symbol: {max_traded_symbol}, Total Traded Value: {max_total_traded_value}")


# In[22]:


import pandas as pd

data = { 'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'), 'Close': [100, 101, 102, 98, 96],
         'Volume': [200, 220, 210, 190, 180], 
}

df = pd.DataFrame(data)

df.set_index('Date', inplace=True)

selected_data = df[(df['Close'] > 100) & (df['Volume'] > 200)]


# In[23]:


stocks_data = {
    'Symbol': ['AAPL', 'MSFT', 'AAPL', 'GOOG', 'MSFT'],
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
    'Close': [150, 250, 145, 1000, 255],
    'Volume': [100, 300, 90, 150, 350]
}
stocks = pd.DataFrame(stocks_data)

stocks['Date'] = pd.to_datetime(stocks['Date'])

selected_stocks = stocks[(stocks['Symbol'] == 'AAPL') & (stocks['Close'] > 140)]

print("Stocks DataFrame:")
print(stocks)
print("\nSelected Rows where 'Symbol' is 'AAPL' and 'Close' is greater than 140:")
print(selected_stocks)


# In[24]:


stocks.sort_values(['Symbol', 'Date'], inplace=True)

stocks['Volume Change'] = stocks.groupby('Symbol')['Volume'].diff().fillna(0)

median_close_price = stocks['Close'].median()
selected_rows = stocks[(stocks['Volume Change'] > 50) & (stocks['Close'] < median_close_price)]

print("Extended Stocks DataFrame:")
print(stocks)
print("\nSelected Rows based on Complex Conditions:")
print(selected_rows)


# In[29]:


import pandas as pd

df_2022 = pd.DataFrame({
    'Date': ['2022-12-29', '2022-12-30'],
    'Stock': ['AAPL', 'AAPL'], 
    'Close': [175, 180]
})
              
df_2023 = pd.DataFrame({
    'Date': ['2023-01-02', '2023-01-03'], 
    'Stock': ['AAPL', 'AAPL'],
    'Close': [178, 182]
})

prices_combined = pd.concat([df_2022 , df_2023], axis=0)


# In[ ]:




