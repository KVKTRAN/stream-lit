import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import matplotlib.pyplot as plt

import yfinance as yf 
import talib
import ta

def cal_target(value, number=1):
    target = []
    for i in range(len(value) - number):
        next_val = value[i+1]
        score = (next_val - value[i]) / value[i]
        target.append(score)
    for i in range(number):
        target.append(0)
    return target
    
def convert_psar(value):
    new_val = []
    
    for i in value:
        if i == 0:
            new_val.append(i)
        else:
            new_val.append(1)
    return new_val

# Intro
st.header("Simple example idea of how to predict stock prices")

# Import data
data = yf.download("AAPL", )
data = data.tail(1000)
data = data.reset_index()

psar = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'])

macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

atr = talib.ATR(high=data["High"], low=data["Low"], close=data["Close"], timeperiod=14)
natr = talib.NATR(high=data["High"], low=data["Low"], close=data["Close"], timeperiod=14)

rsi = talib.RSI(data["Close"])
slowk, slowd = talib.STOCH(high=data["High"], low=data["Low"], close=data["Close"])

adosc = talib.ADOSC(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"])

st.markdown(
    """
    Let's look at the data. As you can see we have time series data from the stock market of Apple company. 
    What we will try to do is predict the rate of change of the stock in the next day using techincal analysis
    """
)
st.write(data)

# Apply technical analysis and rate of change
# Apply techincal analysis
data["psar_up"] = psar.psar_up()
data["psar_down"] = psar.psar_down()

data["macd"] = macd
data["macdsignal"] = macdsignal
data["macdhist"] = macdhist

data["atr"] = atr
data["natr"] = natr

data["rsi"] = rsi
data["slowk"] = slowk
data["slowd"] = slowd

data["adosc"] = adosc

# Cleanup the data
data = data.drop(axis=1, columns=["Date", "Open", "High", "Low", "Volume", "Adj Close"])
data = data.fillna(value={"psar_up": 0, "psar_down":0})
data = data.dropna()
data = data.round(2)

value = data["Close"].to_numpy()
new_psar_up = data["psar_up"].to_numpy()
new_psar_down = data["psar_down"].to_numpy()

target = cal_target(value)
data["Target"] = target
data["psar_up"] = convert_psar(new_psar_up)
data["psar_down"] = convert_psar(new_psar_down)
data.drop(data.tail(1).index,inplace=True) 

data = data.drop(axis=1, columns=["Close"])

st.header("Apply techincal indicators and clean up")
st.markdown(
    """
    We have calculate all the technical indicators that we want to use and clean up the data a little bit
    You can see the Target column, so that is the rate of change that we want to predict. 
    The definition of it is we compare the closing price of today with the next day to see how it has change in percentage.

    So this is the data that we have so far.
    """
)
st.write(data)

# Normalization
st.header("Normalization")
st.markdown(
    """
    We will need to normalize the data, so that the model can predict correctly. 
    
    Take a look at the first row in the table. 
    """
)

normalizer = tf.keras.layers.Normalization(axis=-1)

train_features = data.copy()
test_features = data.copy()
train_labels = train_features.pop('Target')
test_labels = test_features.pop("Target")

normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])

st.markdown("""
    So this is how the first row data looks like, and as you can see some value is very small while the other can be very large.
    Depend on the columns that they are in. It's not a good practice to fit to the model this raw data. 
    So what we will do we will normalize the data
""")
st.write(first)
st.markdown(
    """
    Here is how the data looks like after normalization:
    """
)
st.write(normalizer(first).numpy())

# Train the model
st.header("Build and train the model")
st.markdown(
    """
    So I have built and trained the model, now we will run some prediction
    
    Now we will try to predict the value of the first 10 rows. Let see how it's work
    In good practice we actually have to train test split the data so that we will have the
    training and testing data seperate. 
    
    Prediction:
    """
)
model = tf.keras.models.load_model('stock_model.h5')

value = np.array(test_features.head(10))
test = np.array(test_labels.head(10))

prediction = model.predict(value)
prediction = prediction.flatten()

new_df = pd.DataFrame()
new_df["Predict"] = prediction
new_df["Actual"] = test


st.write(new_df)

# Conclusion

st.header("Conclusion")
st.markdown(
    """
    As you can see apply regression to stock data doesn't work at all, so we need better approach. 
    - Maybe try pattern recognition for the stock prices pattern
    - Candlestick pattern for prediction
    - Add more features 
    - Apply time series model
    """
)