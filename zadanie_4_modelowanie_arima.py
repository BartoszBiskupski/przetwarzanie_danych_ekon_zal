import yfinance as yf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import itertools
import matplotlib.backends.backend_pdf
import datetime

# Get the current date and time
now = datetime.datetime.now()

# Format the current date and time as a string
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

# Create a PdfPages object with the current timestamp in the file name
pdf_pages = matplotlib.backends.backend_pdf.PdfPages(f'zadanie_4_{timestamp_str}.pdf')

# Define a function to download stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

appl_data = download_stock_data('AAPL', '2019-01-01', '2024-05-31')


print(appl_data)
def plot_stock_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.title('AAPL Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the figure to the PDF file
    pdf_pages.savefig(plt.gcf())
    plt.close()

plot_stock_data(appl_data)

# decompose the time series into trend, seasonal, and residual components
def decompose_time_series(df):
    decomposition = sm.tsa.seasonal_decompose(df['Close'], model='multiplicative', period=252)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

aapl_trend, aapl_seasonal, aapl_residual = decompose_time_series(appl_data)
def plot_decomposition(aapl_trend, aapl_seasonal, aapl_residual):
    plt.figure(figsize=(10, 6))
    plt.plot(aapl_trend, label='Trend')
    plt.title('AAPL Stock Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(aapl_seasonal, label='Seasonal')
    plt.title('AAPL Stock Price Seasonal Component')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(aapl_residual, label='Residual')
    plt.title('AAPL Stock Price Residual Component')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# remove non-stationarity from the time series
def remove_non_stationarity(df=appl_data):
    df['Close_diff'] = df['Close'].diff()
    df.dropna(inplace=True)
    return df


def test_stationarity(df):
    result = statsmodels.tsa.stattools.adfuller(df['Close_diff'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])


def test_stationarity_kpss(df):
    result = statsmodels.tsa.stattools.kpss(df['Close_diff'])
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[3])



def plot_acf_pacf(df):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(df['Close_diff'], lags=40, ax=ax[0])
    plot_pacf(df['Close_diff'], lags=40, ax=ax[1])
    plt.tight_layout()
    plt.show()


def fit_arima_model(df, order=(5, 1, 0)):
    model = statsmodels.tsa.arima.model.ARIMA(df["Close"],
                                              order=order)
    results = model.fit()  # Increase the number of iterations
    return results


def forecast_arima_model(df, best_pdq):
    df.index = pd.date_range(start=df.index[0], periods=len(df.index), freq='B')
    forecast = fit_arima_model(df, order=best_pdq).get_forecast(steps=30)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return conf_int, forecast_values


def plot_forecast(df, forecast_values, conf_int):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(forecast_values, label='Forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.2)
    plt.title('AAPL Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf_pages.savefig(plt.gcf())
    plt.close()



# print(conf_int)
# print(forecast_values)



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(1, 5)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Run a grid with pdq parameters calculated
best_aic = np.inf
best_pdq = None
temp_model = None

# conf_int, forecast_values = forecast_arima_model(appl_data, best_pdq)

# plot_forecast(appl_data, forecast_values, conf_int)




# Create a PdfPages object


for param in pdq:
    try:
        temp_model = ARIMA(appl_data["Close"], order=param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param

    except:
        continue

msg = f"Best ARIMA model order is {best_pdq} with AIC: {best_aic}"
conf_int, forecast_values = forecast_arima_model(appl_data, best_pdq)

# Create a new figure
fig, ax = plt.subplots()

# Plot the forecast
plot_forecast(appl_data, forecast_values, conf_int)

# Add the text to the figure
plt.text(0.5, 0.5, msg,
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

# Save the figure to the PDF file
pdf_pages.savefig(fig)

# Close the figure to free up memory
plt.close(fig)

# Close the PDF file
pdf_pages.close()

print(msg)