#!/usr/bin/env python
# coding: utf-8

# In[1]:


dates = ["1st January", "2nd January", "3rd January"]
stock_prices = [150, 152, 151]
for date, price in zip(dates, stock_prices):
    print(f"{date} : ${price}")


# In[4]:


dates = ["4th January", "5th January", "6th January"]
stock_prices = [155, 156, 153]

def calculate_average(prices):
    return sum(prices) / len(prices)

average_price = calculate_average(stock_prices)
print(f"Average Stock Price: ${average_price}")


# In[7]:


def highest_stock(dates, stock_prices):
        max_index = stock_prices.index(max(stock_prices))
        highest_stock_day = dates[max_index]
        return highest_stock_day
    
dates = ["4th January", "5th January", "6th January"]
stock_prices = [155, 156, 153]

result = highest_stock(dates, stock_prices)

print(f"the day is:{result}")


# In[9]:


dates.extend(["7th January", "8th January"])
stock_prices.extend([157, 152])

def analyze_stock_trend(dates, stock_prices):
    increasing = 0 
    decreasing = 0
    stable = 0

    for i in range(1, len(stock_prices)):
        if stock_prices[i] > stock_prices[i - 1]:
            increasing += 1
        elif stock_prices[i] < stock_prices[i - 1]:
            decreasing += 1
        else:
            stable += 1
            
    if increasing > decreasing and increasing > stable:
        return "The stock prices are generally increasing."
    elif decreasing > increasing and decreasing > stable:
        return "The stock prices are generally decreasing."
    else:
        return "The stock prices are generally stable."
    
result = analyze_stock_trend(dates, stock_prices)

print(result)


# In[10]:


dates = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
stock_prices = [150, 152, 151, 153, 152]

import statistics 

def calculate_volatility(prices):
    return statistics.stdev(prices)

volatility = calculate_volatility(stock_prices)
print(f"Volatility: ${volatility}")


# In[11]:


def calculate_average(stock_prices):
    average_price = sum(stock_prices) / len(stock_prices)
    above_average_days = [day for day, price in zip(dates, stock_prices) if price > average_price]
    
    return average_price, above_average_days

average_price, above_average_days = calculate_average(stock_prices)

print(f"Average Stock Price: {average_price}")
print("Days with Stock Price Above Average:", above_average_days)


# In[12]:


def forecast_next_day_price(stock_prices):
    daily_changes = [stock_prices[i + 1] - stock_prices[i] for i in range(len(stock_prices) - 1)]
    average_daily_change = sum(daily_changes) / len(daily_changes)
    
    last_known_price = stock_prices[-1]
    forecasted_price = last_known_price + average_daily_change

    return forecasted_price

stock_prices = [154, 150, 158, 165, 152]

forecasted_price = forecast_next_day_price(stock_prices)

print(f"Forecasted Next Day's Stock Price: {forecasted_price}")


# In[17]:


def present_value(fv, r, n):
    return fv / (1+r)**n

fv = 110
r = 0.1
n = 1

PV = present_value(fv, r, n)
print(f"The present value is: ${PV:.2f}")


# In[18]:


def future_value(pv, r, n):
    return pv * (1 + r)**n

PV = 100
r = 0.1
n=1

FV = future_value(PV, r, n)
print(f"The future value is: ${FV:.2f}")


# In[19]:


def compound(pv, r):
    return pv * (1 + r)

PV = 100
r = 0.1

FV = compound(PV, r)
print(f"After one year with a 10% interest rate, you’ll have: ${FV:.2f}")


# In[20]:


def discount(fv, r):
    return fv / (1 + r)

FV = 110
r = 0.1

PV = discount(FV, r)
print(f"The present value of $110 after one year with a 10% interest rate is: ${PV:.2f}")


# In[23]:


def present_value(fv, r, n):
    return fv / (1 + r)**n 

fv=120
r=0.05
n=2

PV = present_value(fv, r, n)
print(f"the present value is: ${PV:.2f}")


# In[24]:


# Working with Time Series using pandas


# In[25]:


get_ipython().system('pip install pandas yfinance')


# In[26]:


get_ipython().system('pip install pandas yfinance')


# In[27]:


import yfinance as yf

apple_data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

print(apple_data.head())


# In[29]:


import matplotlib.pyplot as plt

apple_data['Close'].plot(figsize=(10, 5))
plt.title('Apple Stock Closing Prices')
plt.ylabel('Price (in \$)')
plt.xlabel('Date')
plt.show()


# In[31]:


apple_data['50-day MA'] = apple_data['Close'].rolling(window=50).mean()
apple_data[['Close', '50-day MA']].plot(figsize=(10, 5))
plt.title('Apple Stock Prices with 50-day Moving Average')
plt.ylabel('Price (in \$)')
plt.xlabel('Date')
plt.show()


# In[35]:


weekly_data = apple_data['Close'].resample('W').mean()
weekly_data.plot(figsize=(10, 5))
plt.title('Apple Stock Weekly Closing Prices')
plt.ylabel('Price (in \$)')
plt.xlabel('Date')
plt.show()


# In[39]:


msft_data = yf.download("MSFT", start= "2021-01-01", end= "2021-12-31")
print(msft_data.head())


# In[41]:


google_data = yf.download("GOOGL", start = "2020-01-01", end = "2023-12-31")
print(google_data.head())


# In[42]:


amazon_data = yf.download("AMZN", start = "2021-10-01", end = "2021-12-31")
print(amazon_data.head())


# In[46]:


tesla_data = yf.download("TSLA", start="2020-01-01", end="2021-01-01")
tesla_data['Close'].plot(figsize=(10, 5))
plt.title('tesla stock closing prices 2020')
plt.ylabel('price (in \$)')
plt.xlabel('date')
plt.show()


# In[50]:


netflix_data = yf.download("NFLX", start="2022-01-01", end="2022-06-30")
netflix_data['Close'].plot(figsize=(10, 5))
plt.title('Netflix closing price for the first half 2022')
plt.ylabel('price (in \$)')
plt.xlabel('date')
plt.show()


# In[ ]:




