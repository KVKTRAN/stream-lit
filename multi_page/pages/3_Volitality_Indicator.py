import pandas as pd 
import numpy as np 
import streamlit as st 
import yfinance as yf 
import matplotlib.pyplot as plt
import ta

import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Technical Analysis"
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

# input data
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

# Setup technical indicator
# atr = talib.ATR(high=data["High"], low=data["Low"], close=data["Close"], timeperiod=14)
# natr = talib.NATR(high=data["High"], low=data["Low"], close=data["Close"], timeperiod=14)
atr = ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"])
atr = atr.average_true_range()


######################################
# Next part is creating chart with technical data analysis included 
# Trend 
st.title("Stock technical analysis")
st.header("Volitality Indicator")
st.markdown(
    """
    Measure the rate of price movement, regardless of direction. This is generally base on a change of in
    the highest and lowest of historical price
    - Average True Range
    - Normalized Average True Range
    """
)

# Show ATR
st.header("Average True Range")
atr_fig = go.Figure()
atr_fig.add_trace(go.Scatter(x=data["Date"], y=atr, name="ATR"))

fig_setting(atr_fig, y_from=min(atr), y_to=max(atr), x_from=0, x_to=len(atr))
st.plotly_chart(atr_fig, use_container_width=True)

# Show NATR
# st.header("Normalized Average True Range")
# natr_fig = go.Figure()
# natr_fig.add_trace(go.Scatter(x=data["Date"], y=natr, name="ATR"))

# # fig_setting(natr_fig, y_from=min(natr), y_to=max(natr), x_from=min_x, x_to=max_x)
# st.plotly_chart(natr_fig, use_container_width=True)