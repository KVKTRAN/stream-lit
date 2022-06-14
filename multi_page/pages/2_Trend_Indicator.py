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

# Setup technical indicator
# Bollinger bands
bb = ta.volatility.BollingerBands(close=data['Close'])
upperband = bb.bollinger_hband()
middleband = bb.bollinger_mavg()
lowerband = bb.bollinger_lband()
# upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# parabolic
# parabolic = talib.SAR(high=data["High"], low=data["Low"])
psar = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'])
parabolic = psar.psar()


# macd
macd = ta.trend.MACD(close=data['Close'])
macd_value = macd.macd()
macdsignal = macd.macd_signal()
macdhist = macd.macd_diff()
# macdsignal, macdhist 

######################################
# Next part is creating chart with technical data analysis included 
# Trend 
st.title("Stock technical analysis")
st.header("Trends Indicators")
st.markdown(
    """
    Measure the direction and strength of a trend, using some form of prices averaging to establish base line
    - Bollinger bands is a famous technical indicators that can be used for volatility, trend,  and momentum analysis.
        - The middle line is the moving average of 20 time periods
        - The upper and lower bands can indicate trends and squezze 
    - Parabolic Stop and Reverse is a very strong trend indicator that can predict very accurate stock trend
    - MACD, an old time indicator that has been used for decades, still very useful until noow
        - Calculate by the two lines from the moving average of a slow period and a fast period
        - The histogram shows the different between the two lines
    """
)

# Show Bollinger Bands
st.header("Bollinger Bands")
bb_fig = fig

bb_fig.add_trace(
    go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'], 
        name="Candle"
    )
)
bb_fig.add_trace(go.Scatter(x=data['Date'], y=upperband, mode="lines", name="Upper BB"))
bb_fig.add_trace(go.Scatter(x=data['Date'], y=middleband, mode="lines", name="Middle BB"))
bb_fig.add_trace(go.Scatter(x=data['Date'], y=lowerband, mode="lines", name="Lower BB"))
min_y -= 5
max_y += 5
fig_setting(bb_fig, y_from=min_y, y_to=max_y, x_from=min_x, x_to=max_x)
st.plotly_chart(bb_fig, use_container_width=True)

# Show only parabolic and stock data
st.header("Parabolic Stop and Reverse")

p_fig = go.Figure(
    data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'], 
    name="Candle")]
)
p_fig.add_trace(go.Scatter(x=data['Date'], y=parabolic, mode="markers", name="Parabolic"))
fig_setting(p_fig, y_from=min_y, y_to=max_y, x_from=min_x, x_to=max_x)
st.plotly_chart(p_fig, use_container_width=True)

# MACD
st.header("MACD")

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=data['Date'], y=macd_value, mode="lines", name="macd"))
macd_fig.add_trace(go.Scatter(x=data['Date'], y=macdsignal, mode="lines", name="macd signal "))
macd_fig.add_trace(go.Bar(x=data['Date'], y=macdhist, name="macd hist"))

fig_setting(macd_fig, y_from=min(macd_value) - 2, y_to=max(macd_value) + 2, x_from=min_x, x_to=max_x)
st.plotly_chart(macd_fig, use_container_width=True)