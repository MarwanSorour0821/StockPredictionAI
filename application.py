#IMPORTS
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

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
start = datetime(end.year - 1, end.month, end.day - 1)


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

#============================================================= Calculating Risk of Stock ================================================================#
#We will use standard deviation to measure the risk because it is a commonly used measure of risk in financial investments
#This is because it will show us how much the returns of the stock deviate from the expected value

rets = tech_rets.dropna()
area = np.pi * 20


plt.figure(figsize=(10,8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
plt.show()

#================================================================ Training AI Model =====================================================================+#
# print("predicting stock price of apple")
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

# plt.figure(figsize=(16,6))
# plt.title("Closing History of Apple")
# plt.xlabel("Date", fontsize=18)
# plt.ylabel("Closing Price USD ($)", fontsize=18)
# plt.show()

data = df.filter(['Close'])
data_set = data.values

#Get number of rows to train model on
training_data_length = int(np.ceil(len(data_set) * 0.95))

# #Scaling the data
# #MinMaxScaler is used to scale the data between 0 and 1. This is used to train many AI models to ensure that all data have the same scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

# # Create the training data set 
# # Create the scaled training data set
train_data = scaled_data[0:int(training_data_length), :]
x_train = []
y_train = []

# #Tested on 60 historical trading days 
# for i in range(60, len(train_data)):
#     #appends a sequence of historical closing prices to the x_train list. The sequence is extracted from the train_data array, 
#     #starting from index i-60 (60 days before the current day i) up to index i-1 (the day just before the current day i). 
#     #The [:, 0] indexing selects only the closing prices from this sequence. So essentially, x_train is a list of sequences, 
#     #each containing 60 historical closing prices.
#     x_train.append(train_data[i-60:i, 0])
#     y_train.append(train_data[i, 0])
#     if i<= 61:
#         print(x_train)
#         print(y_train)
#         print()

# # Convert the x_train and y_train to numpy arrays 
# x_train, y_train = np.array(x_train), np.array(y_train)

# # Reshape the data into a 3 dimensional array since many machine learnign libraries require this format
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# #TRAIN THE MODEL ON  TRAIN SET
# #Build LSTM Model
# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(x_train, y_train, batch_size=1, epochs=1)

# #============================================================= BUIlD TESTING DATA ========================================================================#

# # Create the testing data set
# # Create a new array containing scaled values from index 1543 to 2002 
# test_data = scaled_data[training_data_length - 60: , :]
# # Create the data sets x_test and y_test
x_test = []
# y_test = data_set[training_data_length:, :]
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i, 0])

# #convert to numpy array
x_test = np.array(x_test)

# #Reshape to 3 dimensional array
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)

# #Get root mean square
# rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# # Plot the data
# train = data[:training_data_length]
# valid = data[training_data_length:]
# valid['Predictions'] = predictions
# # Visualize the data
# plt.figure(figsize=(16,6))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# print(valid[['Close', 'Predictions']])

# Adjusting hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 64

print("FFF")

# Increase training data length
training_data_length = int(np.ceil(len(data_set) * 0.98))  # Using 98% of the data for training

test_data = scaled_data[training_data_length - 60:, :]
x_test = []
y_test = data_set[training_data_length:, :]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :])  # Including all features
    y_train.append(train_data[i, 0])  # Target remains the same (closing price)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Adjusting model architecture
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Adding dropout layer for regularization
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))

# Compiling the model with the adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Training the model with more epochs
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

print("GGGG")

# for i in range(60, len(train_data)):
#     x_train.append(train_data[i-60:i, :])  # Including all features
#     y_train.append(train_data[i, 0])  # Target remains the same (closing price)

# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# # Retraining the model with updated input data
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

# Getting testing data
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Making predictions with the updated model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculating RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plotting results
train = data[:training_data_length]
valid = data[training_data_length:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

print(valid[['Close', 'Predictions']])
