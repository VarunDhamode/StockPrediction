# import alpaca_trade_api as tradeapi
#
# # Initialize Alpaca API client
# api = tradeapi.REST(api_key='your_api_key', api_secret='your_api_secret', base_url='https://paper-api.alpaca.markets')
#
# # Get indicator data
# indicator_data = api.get_indicator(symbol='AAPL', indicator='sma', period='1d', time_range='1mo')
#
# # Process indicator data
# # (Parse data, convert to desired format, etc.)
#
# # Use indicator data in your trading strategy
# # (Make buy/sell decisions, generate signals, etc.)
import keras
import numpy as np
import requests
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import keras
import pandas as pd
from random2 import sp500_stocks, nifty50_stocks,top_100_crypto_list
# import streamlit as st

import streamlit as st
import random
import string


import streamlit as st
import random
import string
import datetime

current_date = datetime.datetime.now().date()

# Define your functions and imports here

StockName = None




st.set_page_config(page_title='Stock Trend Prediction', page_icon='📈',
                   layout="centered", initial_sidebar_state="expanded")

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}


def generate_random_key(length=6):
    """Generate a random key of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)

st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ['Stock Prediction', 'Stock Analysis',"Stock News"])

if page == 'Stock Prediction':
    st.markdown("<center><h1 style='color: green;'>Stock Trends Prediction 📈</h1></center>",
                unsafe_allow_html=True)
    st.markdown(
        '<center><h3>A Time Series analogy LSTM Deep Learning Model</h3></center>', unsafe_allow_html=True)
    listStocks = ["Nifty50", "S&P500", "Crypto"]
    select = st.selectbox("which stocks you want to search", listStocks)
    if select == "S&P500":
        stock_ticker_ip = st.selectbox("Type or Select Stock of your choice", sp500_stocks)
    elif select == "Nifty50":
        stock_ticker_ip = st.selectbox("Type or Select Stock of your choice", nifty50_stocks)
    else:
        stock_ticker_ip = st.selectbox("Type or Select Stock of your choice", top_100_crypto_list)

    st.session_state.stock_data['stock_ticker'] = stock_ticker_ip




    # Load the scaler and model if not already loaded
    if 'scaler' not in st.session_state.stock_data:
        st.session_state.stock_data['scaler'] = MinMaxScaler(feature_range=(0, 1))

    if 'model' not in st.session_state.stock_data:
        st.session_state.stock_data['model'] = keras.models.load_model('keras_stock_prediction_model.keras')


    # Rest of your prediction page code goes here
    data = yf.download(stock_ticker_ip, start='2015-01-01', end=current_date)
    # data = yf.download(sp500_stocks, start=start_timestamp, end=end_timestamp)

    # describing the model
    st.markdown("---")
    st.markdown('<h3>Data from 2010-2024</h3>',
                unsafe_allow_html=True)
    st.subheader("Part of a Data")
    st.dataframe(data.head().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'white')]}
    ]))
    # st.write(data.describe())
    st.subheader("Complete Describition of Data")
    st.dataframe(data.describe().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'white')]}
    ]))

    # visualisations
    st.markdown("---")
    st.markdown('<center><h3>Closing Price vs Time chart</h3></center>',
                unsafe_allow_html=True)
    fig = plt.figure(figsize=(9, 5))
    plt.plot(data.Close, 'g', label='Closing price')
    plt.xlabel('Years', fontsize=13)
    plt.ylabel('Closing Price', fontsize=13)
    plt.legend()
    st.pyplot(fig)
    st.markdown("---")

    st.markdown('<center><h3>100 days Moving Average</h3></center>',
                unsafe_allow_html=True)
    moving_avg100 = data.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(data.Close, 'b', label='Closing price')
    plt.plot(moving_avg100, 'g', label='Moving Average 100 days')
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Closing Price', fontsize=14)
    plt.legend()
    st.pyplot(fig)
    st.markdown("---")

    st.markdown('<center><h3>100 and 200 days Moving Average</h3></center>',
                unsafe_allow_html=True)
    moving_avg100 = data.Close.rolling(100).mean()
    moving_avg200 = data.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(moving_avg100, 'r', label='Moving Average 100 days')
    plt.plot(moving_avg200, 'g', label='Moving Average 200 days')
    plt.plot(data.Close, 'b', label='Closing price')
    plt.legend()
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    st.pyplot(fig)
    st.markdown("---")

    data_training = pd.DataFrame(data['Close'][0: int(len(data) * 0.7)])
    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # load the model
    model = keras.models.load_model('keras_stock_prediction_model.keras')

    # testing
    prev_100_days = data_training.tail(100)
    new_df = pd.concat([prev_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(new_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scale = scaler.scale_
    factor = 1 / scale[0]
    y_test = y_test * factor
    y_predicted = y_predicted * factor

    # Chart of 100 MA, 200MA and closing price
    st.markdown('<center><h3>Predictions vs Original</h3></center>',
                unsafe_allow_html=True)
    fig = plt.figure(figsize=(12, 6))
    plt.title('Original vs Predicted Price Graph', fontsize=15)
    plt.plot(y_test, 'b', label='Original price')
    plt.plot(y_predicted, 'r', label='Predicted price')
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    st.pyplot(fig)
    st.markdown("---")

    st.write("<center><h5><span style='color: white;'>~ Made with ❤️ by Varun Dhamode</span><h5></center>",
             unsafe_allow_html=True)

elif page == 'Stock Analysis':
    def predict_future_trend(stock_ticker):
        # Load the data for prediction
        data = yf.download(stock_ticker, start='2015-01-01', end=current_date)
        data_training = pd.DataFrame(data['Close'][0: int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        # Prepare data for scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load the model
        model = keras.models.load_model('keras_stock_prediction_model.keras')

        # Prepare the last 100 days data for prediction
        prev_100_days = data_training.tail(100)
        new_df = pd.concat([prev_100_days, data_testing], ignore_index=True)

        # Select only the 'Close' column for transformation
        input_data = scaler.transform(new_df[['Close']])

        x_test = []

        # Using the last 100 days data for prediction
        x_test.append(input_data[-100:])

        x_test = np.array(x_test)

        # Predict the next 20 days trend
        predictions = []
        for i in range(20):
            # Predict the next day
            prediction = model.predict(x_test)
            predictions.append(prediction[0][0])

            # Update x_test for the next prediction
            x_test = np.roll(x_test, -1, axis=1)
            x_test[0][-1] = prediction[0][0]

        # Generate future dates for the next 20 days
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=21)[1:]  # Exclude the current date

        return predictions, future_dates


    def stock_analysis(stock_ticker):
        st.markdown("<center><h1 style='color: blue;'>Stock Analysis 📊</h1></center>",
                    unsafe_allow_html=True)

        # Load the data for prediction
        data = yf.download(stock_ticker, start='2015-01-01', end=current_date)
        data_training = pd.DataFrame(data['Close'][0: int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        # Prepare data for scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load the model
        model = keras.models.load_model('keras_stock_prediction_model.keras')

        # Prepare the last 100 days data for prediction
        prev_100_days = data_training.tail(100)
        new_df = pd.concat([prev_100_days, data_testing], ignore_index=True)

        # Select only the 'Close' column for transformation
        input_data = scaler.transform(new_df[['Close']])

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        # Calculate factor
        scale = scaler.scale_
        factor = 1 / scale[0]

        # Plot predictions vs original
        st.markdown('<center><h3>Predictions vs Original</h3></center>',
                    unsafe_allow_html=True)
        fig = plt.figure(figsize=(12, 6))
        plt.title('Original vs Predicted Price Graph', fontsize=15)
        plt.plot(y_test * factor, 'b', label='Original price')
        plt.plot(y_predicted * factor, 'r', label='Predicted price')

        plt.xlabel('Days', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend()
        st.pyplot(fig)

        # Get the predicted trend for the next 20 days
        predictions, future_dates = predict_future_trend(stock_ticker)

        # Plot the predicted trend for the next 20 days in a separate graph
        st.markdown('<center><h3>Predicted Trend for the Next 20 Days</h3></center>',
                    unsafe_allow_html=True)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(future_dates, np.array(predictions) * factor, 'g--', label='Predicted trend (Next 20 days)')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend()
        st.pyplot(fig)


    listStocks = ["Nifty50", "S&P500", "Crypto"]
    select = st.sidebar.selectbox("Select Stock", listStocks)
    if select == "S&P500":
        stock_ticker_ip = st.sidebar.selectbox("Select Stock Ticker", sp500_stocks)
    elif select == "Nifty50":
        stock_ticker_ip = st.sidebar.selectbox("Select Stock Ticker", nifty50_stocks)
    else:
        stock_ticker_ip = st.sidebar.selectbox("Select Stock Ticker", top_100_crypto_list)

    stock_analysis(stock_ticker_ip)
    # future 20 days prediction
    st.markdown("<center><h2 style='color: blue;'>Predicted Future Trend for the Next 20 Days</h2></center>",
                unsafe_allow_html=True)
    predictions = predict_future_trend(stock_ticker_ip)
    st.write(predictions)

elif page == "Stock News":
    # Constants
    NEWS_API_KEY = 'dd8c8b387c8c4252b62a6bceb6c3b138'  # Replace with your NewsAPI key
    NEWS_API_URL = 'https://newsapi.org/v2/everything'
    def fetch_news(query, api_key=NEWS_API_KEY):
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt'
        }
        response = requests.get(NEWS_API_URL, params=params)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            st.error('Failed to fetch news')
            return []


    def display_news(articles):
        for article in articles:
            st.subheader(article['title'])
            if article['author']:
                st.write(f"**Author:** {article['author']}")
            if article['source']['name']:
                st.write(f"**Source:** {article['source']['name']}")
            if article['publishedAt']:
                st.write(f"**Published at:** {article['publishedAt']}")
            if article['url']:
                st.write(f"**Read more:** [link]({article['url']})")
            if article['urlToImage']:
                st.image(article['urlToImage'], use_column_width=True)
            st.write(article['description'])
            st.write('---')


    st.title('Stock Analysis App')

    # Define the buttons for fetching news
    if st.button('Get Latest Forex News'):
        st.experimental_set_query_params(page='forex_news')
        st.experimental_rerun()

    if st.button('Get Latest Crypto News'):
        st.experimental_set_query_params(page='crypto_news')
        st.experimental_rerun()

    query_params = st.experimental_get_query_params()
    page = query_params.get('page', [None])[0]

    if page == 'forex_news':
        st.header('Latest Forex News')
        articles = fetch_news('forex')
        display_news(articles)

    elif page == 'crypto_news':
        st.header('Latest Crypto News')
        articles = fetch_news('crypto')
        display_news(articles)


