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
acc = ta.volume.AccDistIndexIndicator(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"])
acc_index = acc.acc_dist_index()
obv = ta.volume.OnBalanceVolumeIndicator(data["Close"], data["Volume"])
obv = obv.on_balance_volume()

######################################
# Next part is creating chart with technical data analysis included 
# Trend 
st.title("Stock technical analysis")
st.header("Volume Indicator")
st.markdown(
    """
    Measure the strength of a trend or confirm a trading direction based on some form of 
    averaging or smoothing of raw volume.
    - On Balance Volume
    - Chaikin A/D Oscillator
    """
)

# Show OBV
st.header("On Balance Volume")
obv_fig = go.Figure()
obv_fig.add_trace(go.Scatter(x=data["Date"], y=obv, name="ATR"))

# fig_setting(atr_fig, y_from=min(atr), y_to=max(atr), x_from=min_x, x_to=max_x)
st.plotly_chart(obv_fig, use_container_width=True)

# ACC
st.header("Chaikin A/D Oscillator")
acc_fig = go.Figure()
acc_fig.add_trace(go.Scatter(x=data["Date"], y=acc_index, name="ACC"))

# fig_setting(natr_fig, y_from=min(natr), y_to=max(natr), x_from=min_x, x_to=max_x)
st.plotly_chart(acc_fig, use_container_width=True)