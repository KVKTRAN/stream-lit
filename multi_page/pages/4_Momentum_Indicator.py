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
# cci = talib.CCI(high=data["High"], low=data["Low"], close=data["Close"])
# rsi = talib.RSI(data["Close"])
# slowk, slowd = talib.STOCH(high=data["High"], low=data["Low"], close=data["Close"])

cci = ta.trend.CCIIndicator(high=data["High"], low=data["Low"], close=data["Close"])
rsi = ta.momentum.RSIIndicator(close=data["Close"])
stoch = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"])

cci = cci.cci()
rsi = rsi.rsi()
slowk = stoch.stoch()
slowd = stoch.stoch_signal()

######################################
# Next part is creating chart with technical data analysis included 
# Trend 
st.title("Stock technical analysis")
st.header("Momentum Indicator")
st.markdown(
    """
    Help indentify the speed of price movement by comparing prices over time.
    - Commodity Channel Index
    - Relative Strength Index
    - Stochastic Oscillator
    """
)

# Show CCI
st.header("Commodity Channel Index")
cci_fig = go.Figure()
cci_fig.add_trace(go.Scatter(x=data["Date"], y=cci, name="CCI"))

fig_setting(cci_fig, y_from=min(cci), y_to=max(cci), x_from=min_x, x_to=max_x)
st.plotly_chart(cci_fig, use_container_width=True)

# Show RSI
st.header("Relative Strength Index")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=data["Date"], y=rsi, name="RSI"))

fig_setting(rsi_fig, y_from=min(cci), y_to=max(cci), x_from=min_x, x_to=max_x)
st.plotly_chart(rsi_fig, use_container_width=True)

# Stochastic
st.header("Stochastic")
s_fig = go.Figure()
s_fig.add_trace(go.Scatter(x=data["Date"], y=slowk, name="slowk"))
s_fig.add_trace(go.Scatter(x=data["Date"], y=slowd, name="slowd"))


fig_setting(s_fig, y_from=0, y_to=100, x_from=min_x, x_to=max_x)
st.plotly_chart(s_fig, use_container_width=True)