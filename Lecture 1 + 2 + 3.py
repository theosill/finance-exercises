#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np


# In[62]:


ticker = "AAPL"
opening_price = 142.7
closing_price = 143.2
volume = 1200000
print(ticker, opening_price, closing_price, volume)


# In[63]:


Currency_pair = "EUR/USD"
Buying_rate = 1.1825
Seeling_rate = 1.1830
print(Currency_pair, Buying_rate, Seeling_rate)


# In[64]:


liste = ["AAPL", "MSFT", "GOOGL"]
liste.append("IBM")
print(liste)


# In[65]:


stock_details = {
"ticker": "AAPL",
"opening_price": 142.7,
"closing_price": 143.2,
"volume": 1200000
}
print(stock_details)


# In[1]:


bond_details = {
"Issuer": "Théo", 
"Maturity Date": "2023-12-01", 
"Coupon Rate": 4.2,
"Face Value": 120,
}
print(bond_details)


# In[67]:


stock_prices = [100, 101, 102, 98, 97]
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
    print(daily_return)


# In[68]:


principal = 1000
rate = 0.05 #% 5% annual interest
years = 0
while principal < 2000:
    principal *= (1 + rate)
    years += 1
    
print(years)


# In[69]:


stock_prices = [105, 107, 104, 106, 103]
test = []
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
    test.append(daily_return)
    print(daily_return)
    
print(np.mean(test))


# In[70]:


principal = 500
rate = 0.07
years = 0
while principal < 1000:
    principal *= (1 + rate)
    years += 1
print(years)
print(principal)


# In[71]:


bond_yield = 4.5
if bond_yield > 4.0:
    print("Buy the bond.")


# In[72]:


pe_ratio = 20
if pe_ratio < 15:
    print("Buy the stock.")
elif pe_ratio > 25:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[73]:


bond_yield = 3.8
if bond_yield >= 4.0:
    print("Buy the bond.")
else:
    print("Do not buy the bond.")


# In[74]:


pe_ratio = 17
if pe_ratio < 15:
    print("Buy the stock.")
elif pe_ratio > 25:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[75]:


pe_ratio = 15
if pe_ratio < 16 and pe_ratio > 14:
    print("Buy the stock.")
elif pe_ratio > 23 and pe_ratio < 27:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[76]:


class Stock:
        def __init__(self, name, price, dividend):
            self.name = name
            self.price = price
            self.dividend = dividend
        def yield_dividend(self):
            return self.dividend / self.price

apple_stock = Stock('Apple', 150, 0.82)
google_stock = Stock('Google', 150, 0.82)
facebook_stock = Stock('Facebook', 150, 0.82)
print(apple_stock.yield_dividend())


# In[77]:


class Portfolio:
        def __init__(self, name):
            self.name = name
            self.instruments = []
                  
        def add_instrument(self, stock):
            self.instruments.append([stock.name, stock.price, stock.dividend])
        
        def total_value(self):
            test = 0
            for i in range(len(self.instruments)):
                test += self.instruments[i][1]
            return test

portfolio_1 = Portfolio('Portfolio_1')
portfolio_1.add_instrument(apple_stock)
portfolio_1.add_instrument(google_stock)
portfolio_1.add_instrument(facebook_stock)

print(portfolio_1.instruments)


# In[78]:


portfolio_1.total_value()


# In[79]:


class CurrencyConverter:
        def __init__(self):
            self.conversion_rate = {}
                  
        def add_rate(self, pair, rate):
            self.conversion_rate[pair] = rate
            
        def convert(self, amount, source_currency, target_currency):
            return amount * self.conversion_rate[source_currency + "/" + target_currency]
    
                


# In[80]:


currency_converter_1 = CurrencyConverter()
currency_converter_1.add_rate("EUR/USD", 1.05)
currency_converter_1.add_rate("USD/EUR", 0.95)
currency_converter_1.convert(100, "EUR", "USD")


# In[81]:


import numpy as np
prices = np.array([100, 102, 104, 101, 99, 98])
returns = (prices[1:] - prices[:-1]) / prices[:-1]
print("Daily returns:", returns)
annual_volatility = np.std(returns) * np.sqrt(252)
print("Annualized volatility:", annual_volatility)


# In[82]:


import numpy as np

np.random.seed(0)
daily_returns = np.random.normal(0.001, 0.02, 1000)
stock_prices = [100]
for r in daily_returns:
    stock_prices.append(stock_prices[-1] * (1+r))
    
stock_prices


# In[83]:


sigma_1 = 0.1
sigma_2 = 0.2

p1_2 = 0.5

w_1 = 0.6
w_2 = 0.4

var_p = w_1**2 * sigma_1**2 + w_2**2 * sigma_2**2 + 2 * w_1 * w_2 * sigma_1 * sigma_2 * p1_2
print(var_p)


# In[84]:


asset_A = [0.1, 0.2]
asset_B = [0.15, 0.3]

weights_A = np.array([i/10 for i in range(11)])

weights_B = 1 - weights_A

returns = asset_B[0]*weights_B + asset_A[0]*weights_A

print("returns :", returns)

covA_B = 0.5

vola = (weights_B**2 * asset_B[1]**2 + weights_A**2 * asset_A[1]**2 + 2 * weights_A * weights_B * asset_A[1] * asset_B[1] * covA_B)**0.5
print("volatilities :", vola)


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns
stock_prices = [100, 102, 104, 103, 105, 107, 108]
dates = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(10, 6))
sns.lineplot(x=dates, y=stock_prices)
plt.title('Stock Price Over a Week')
plt.xlabel("Days")
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()


# In[86]:


import matplotlib.pyplot as plt
stock_prices = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
plt.plot(stock_prices)
plt.title("Stock Prices Over 10 Days")
plt.xlabel('Days')
plt.ylabel("Stock Price")
plt.show()


# In[87]:


import matplotlib.pyplot as plt
stock_prices_1 = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
stock_prices_2 = [107, 108, 107, 107, 106, 108, 109, 108, 109, 110]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Days')
ax1.set_ylabel('stock price 1', color='red')
ax1.plot(stock_prices_1, color='red', ls="solid")
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('stock price 2', color='blue')
ax2.plot(stock_prices_2, color='blue', ls="dotted")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Stock Prices Over 10 Days')

plt.show()


# In[88]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
sns.histplot(returns, bins=5)
plt.title("Distribution of Stock Returns")
plt.show()


# In[89]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]        
sns.histplot(returns, bins=5, kde=True)
plt.title("Distribution of Stock Returns")
plt.show()


# In[90]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = np.random.normal(0, 1, 10000)
sns.histplot(returns, bins=50, kde=True)
plt.title("Distribution of Stock Returns")
plt.show()


# In[ ]:





# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#finance classes
class Stock:
        def __init__(self, name, price, dividend):
            self.name = name
            self.price = price
            self.dividend = dividend
        def yield_dividend(self):
            return self.dividend / self.price

apple_stock = Stock('Apple', 120, 0.82)
google_stock = Stock('Google', 150, 0.82)
facebook_stock = Stock('Facebook', 130, 0.82)

print("Apple stock yield dividend: ", apple_stock.yield_dividend())

class Portfolio:
        def __init__(self, name):
            self.name = name
            self.instruments = []
                  
        def add_instrument(self, stock):
            self.instruments.append([stock.name, stock.price, stock.dividend])
        
        def total_value(self):
            test = 0
            for i in range(len(self.instruments)):
                test += self.instruments[i][1]
            return test

portfolio_1 = Portfolio('Portfolio_1')
portfolio_1.add_instrument(apple_stock)
portfolio_1.add_instrument(google_stock)
portfolio_1.add_instrument(facebook_stock)

print("Portfolio instruments: ", portfolio_1.instruments)
print("Portfolio total value: ", portfolio_1.total_value())

class CurrencyConverter:
        def __init__(self):
            self.conversion_rate = {}
                  
        def add_rate(self, pair, rate):
            self.conversion_rate[pair] = rate
            
        def convert(self, amount, source_currency, target_currency):
            return amount * self.conversion_rate[source_currency + "/" + target_currency]
    
currency_converter_1 = CurrencyConverter()
currency_converter_1.add_rate("EUR/USD", 1.05)
currency_converter_1.add_rate("USD/EUR", 0.95)
print("100 euros to usd: ", currency_converter_1.convert(100, "EUR", "USD"))

#2-assets portfolio returns and volatilities

asset_A = [0.1, 0.2]
asset_B = [0.15, 0.3]

weights_A = np.array([i/10 for i in range(11)])

weights_B = 1 - weights_A

returns = asset_B[0]*weights_B + asset_A[0]*weights_A

covA_B = 0.5

vola = (weights_B**2 * asset_B[1]**2 + weights_A**2 * asset_A[1]**2 + 2 * weights_A * weights_B * asset_A[1] * asset_B[1] * covA_B)**0.5
print("returns :", returns)
print("volatilities :", vola)

#finance plots
stock_prices_1 = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
stock_prices_2 = [107, 108, 107, 107, 106, 108, 109, 108, 109, 110]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Days')
ax1.set_ylabel('stock price 1', color='red')
ax1.plot(stock_prices_1, color='red', ls="solid")
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('stock price 2', color='blue')
ax2.plot(stock_prices_2, color='blue', ls="dotted")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Stock Prices Over 10 Days')

plt.show()

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
sns.histplot(returns, bins=5)
plt.title("Distribution of Stock Returns")
plt.show()


# In[ ]:





# In[ ]:




