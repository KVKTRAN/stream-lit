import pandas as pd 
import numpy as np 
import streamlit as st 
import yfinance as yf 
import matplotlib.pyplot as plt

import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Data Visualization"
)

def fig_setting(fig, y_from, y_to, x_from=0, x_to=100):
    fig.update_xaxes(
        type='category', 
        showticklabels=False,
    )
    fig.update_layout(
        width=1000,


        xaxis_range=[x_from, x_to],
        yaxis_range=[y_from, y_to],
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
        showlegend=True,
    )

# download data and process data
data = yf.download("AAPL", )
data = data.tail(1000)
data = data.reset_index()

# print data and len
st.title("Demo on stock data visualization")

st.header("Stock data in table")
st.markdown(
    """
    This is stock  price of Apple company. Ticker is AAPL
    """
)
st.write(data)
st.write(data.describe())

# input data
# min_x = st.number_input('Insert a starting number: ', value=len(data) - 70, format="%d", step=1)
# max_x = st.number_input('Insert a ending number: ', value=len(data), format="%d", step=1)

min_x = int(len(data) - 70)
max_x = int(len(data) + 1)

min_y = min(data['Low'][min_x:max_x]) - 5
max_y = max(data['High'][min_x:max_x]) + 5


# generate fig 
fig = go.Figure(
    data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'], 
    name="Candle")]
)

# Plot plotly chart
st.header("Stock data in candlestick chart")

fig_setting(fig, y_from=min_y, y_to=max_y, x_from=min_x, x_to=max_x)
# fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode="lines"),)

st.plotly_chart(fig, use_container_width=True)


# ######################################
# # Next part is creating chart with technical data analysis included 
# # Volatility 

# st.title("Stock technical analysis")

# upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
# fig.add_trace(go.Scatter(x=data['Date'], y=upperband, mode="lines", name="Upper BB"))
# fig.add_trace(go.Scatter(x=data['Date'], y=middleband, mode="lines", name="Middle BB"))
# fig.add_trace(go.Scatter(x=data['Date'], y=lowerband, mode="lines", name="Lower BB"))

# min_y -= 5
# max_y += 5
# fig_setting(fig, y_from=min_y, y_to=max_y, x_from=min_x, x_to=max_x)

# st.plotly_chart(fig, use_container_width=True)