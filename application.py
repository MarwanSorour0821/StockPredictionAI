#IMPORTS
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
from pandas_datareader import data
from datetime import datetime

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

#for stock in tech_list:
 #   globals()[stock] = yf.download(stock,start,end)

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

plt.figure(figsize=(15, 10))
plt.subplots(top = 1.25, bottom = 1.2)

for i, company in enumerate(tech_list):
    plt.subplot(2,2, i)
    company['Adj Close'].plot()
    plt.title(f'Closing of {company}')
    plt.ylabel("Adj Closing Price")
    plt.xlabel(None)

plt.tight_layout()

#Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
#print(msft.history(period="max"))
