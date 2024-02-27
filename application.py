#IMPORTS
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime

#--------------------------------------------------------- GETTING COMPANY DATA -----------------------------------------------------------------#
#sns that set aesthetic style of plots
sns.set_style("whitegrid")

#Set overall visual style of plots
plt.style.use("fivethirtyeight")

#To use yfinance under the hood when trying to retrieve data
yf.pdr_override()

#The list of companies that we want to predict the stocks for
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

#set up start and end dates
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)


company_data = {}
for stock_symbol in tech_list:
    company_data[stock_symbol] = yf.download(stock_symbol, start, end)

# Add a new column to each DataFrame with the company name
#Pairs the stock symbols with the company names using the zip function
for stock_symbol, company_name in zip(tech_list, ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]):
    company_data[stock_symbol]["company_name"] = company_name

# Concatenate the DataFrames into a single DataFrame
df = pd.concat(company_data.values(), keys=tech_list, names=["Stock", "Date"])
df.reset_index(inplace=True)
df.tail(10)

#------------------------------------------------------Plotting Closing Price and Volume -----------------------------------------------------------#
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top = 1.25, bottom = 1.2)


#We use 1 here for the enumerate because subplot takes values only between 1<= and <= 4
for i, company_stock_symbol in enumerate(tech_list, 1):
    plt.subplot(2,2, i)

    #Get Closing data for current compnay symbol
    company_df = company_data[company_stock_symbol]
    company_df['Adj Close'].plot()
    plt.title(f'Closing of {tech_list[i - 1]}')
    plt.ylabel("Adj Closing Price")
    plt.xlabel(None)

plt.tight_layout()
plt.show()

print("Proceeding to volume")

for i, company_stock_symbol in enumerate(tech_list, 1):
    plt.subplot(2,2,i)
    #Get Volume data for current company symbol
    company_df = company_data[company_stock_symbol]
    company_df['Volume'].plot()
    plt.title(f'Total volume traded each day for {tech_list[i-1]}')
    plt.ylabel("Volume Traded Per Day")
    plt.xlabel(None)

plt.tight_layout()
plt.show()

#------------------------------------------------ GETTING MOVING AVERAGE OF EACH STOCK --------------------------------------------------#

ma_day = [10,20,50]

for ma in ma_day:
    for company in  company_data:
        column_name = f'MA - {ma} days'
        company_df = company_data[company]
        company_df[column_name] = company_df['Adj Close'].rolling(ma).mean()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL = company_data['AAPL']
GOOG = company_data['GOOG']
MSFT = company_data['MSFT']
AMZN = company_data['AMZN']

AAPL[['Adj Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(ax=axes[0,1])
axes[0,0].set_title('GOOGLE')

MSFT[['Adj Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(ax=axes[1,0])
axes[0,0].set_title('MICROSOFT')

AMZN[['Adj Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(ax=axes[1,1])
axes[0,0].set_title('AMAZON')

fig.tight_layout()
plt.show()
#Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
#print(msft.history(period="max"))

#=========================================================================Calculating Daily Returns =====================================================#

fig1, axess = plt.subplots(nrows=2, ncols=2)
fig1.set_figheight(10)
fig1.set_figwidth(15)

for company in  company_data:
        #company_df = company_data[company]
        #company_df['Daily Return'] = company_df['Adj Close'].pct_change()*100
    company_df = company_data[company]
    company_df['Adj Close'] = pd.to_numeric(company_df['Adj Close'], errors='coerce')
    company_df['Daily Return'] = company_df['Adj Close'].pct_change() * 100
    company_df.dropna(subset=['Daily Return'], inplace=True)

AAPL['Daily Return'].plot(ax = axess[0,0], legend=True, linestyle='--', marker='o')
axess[0,0].set_title('Apple Daily Returns')
GOOG['Daily Return'].plot(ax=axess[0,1], legend=True, linestyle='--', marker='o')
axess[0,1].set_title('Google Daily Returns')
MSFT['Daily Return'].pct_change().plot(ax=axess[1,0], legend=True, linestyle='--', marker='o')
axess[1,0].set_title('Microsoft Daily Returns')
AMZN['Daily Return'].plot(ax=axess[1,1], legend=True, linestyle='--', marker='o')
axess[1,1].set_title('Amazon Daily Returns')
fig1.tight_layout()
plt.show()

for i, company in enumerate(tech_list, 1):
     plt.subplot(2,2,i)
     company_df = company_data[company]
     plt.hist(company_df['Daily Return'], bins=50)
     plt.xlabel('Daily Returns')
     plt.ylabel('Counts')
     plt.title(f'{tech_list[i - 1]}')
plt.tight_layout()
plt.show()

#============================================================== Pair Correlation of Companies ==========================================================#

closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new DataFrame to store the percentage change of Adj Close
tech_rets = closing_df.pct_change()

sns.jointplot(x='GOOG', y='MSFT', data = tech_rets, kind='scatter', color='seagreen')
plt.show()

sns.pairplot(tech_rets, kind='reg')
plt.show()