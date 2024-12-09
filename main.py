import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pmdarima import auto_arima

def EDA(data):
    # type of columns
    print("Type of columns: ", data.dtypes)
    # Convert date column to datetime
    data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
    print("\nDate column converted to:", data[data.columns[0]].dtype)
    # head of data
    print("Head of data: \n",data.head())
    # min/max values
    print("\nMin values: \n", data.min())
    print("\nMax values: \n", data.max())


    # distribution of values
    ax = data.plot.hist(bins=12, alpha=0.5)
    plt.title("Distribution of Values")
    plt.show()

    # missing values of date
    print("\nMissing values: \n", data.isnull().sum())

    # time series plot
    plt.figure(figsize=(10,6))
    data.plot(x=data.columns[0], y=data.columns[1:])
    plt.title("Time Series Plot")
    plt.xticks(rotation=45)
    plt.show()

    # detect outliers
    print("\nOutliers: \n", data.describe())
    return data

def ARIMA_EDA (data, lags):
    data_og = data.copy()
    # plot initial acf and pacf
    plt.figure(figsize=(10,6))
    plot_acf(data["value"], lags=lags)
    plt.title('Autocorrelation Function (ACF)')
    plt.figure(figsize=(10,6))
    plot_pacf(data["value"], lags=lags)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

    d = 0
    max_d = 2  # Enimmäisarvo d:lle
    while True:
        result = adfuller(data_og["value"])
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        
        if result[1] <= 0.05:  # Jos p-arvo on alle 0.05, sarja on stationaarinen
            print("The series is stationary, with d =", d)
            break
        elif d >= max_d:  # Estetään liiallinen differointi
            print("Reached maximum d =", max_d)
            break
        else:
            # Differoi sarjaa ja kasvata d
            data_og["value"] = data_og["value"].diff()
            data_og = data_og.dropna()
            d += 1
    print("d = ", d)
    
    # determine p and q
    plt.figure(figsize=(10,6))
    plot_acf(data["value"], lags=lags)
    plt.title('Autocorrelation Function (ACF)')
    plt.figure(figsize=(10,6))
    plot_pacf(data["value"], lags=lags)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

def auto_arima_model(data):
    model = auto_arima(
        data["value"],            # Aikasarjadata
        start_p=0,                # Alkuarvo p:lle
        start_q=0,                # Alkuarvo q:lle
        max_p=10,                  # Suurin arvo p:lle
        max_q=10,                  # Suurin arvo q:lle
        m=1,                      # Jaksojen määrä (m=1 ei-kausiluonteiselle datalle)
        seasonal=False,           # Ei-kausiluonteinen malli
        trace=True,               # Tulosta tiedot konsoliin
        error_action='ignore',    # Ohita virheet
        suppress_warnings=True,   # Poista varoitukset
        stepwise=True             # Käytä stepwise-menetelmää nopeuttamiseen
    )
    print(model.summary())
    return model

def ARIMA_model(data, p, d, q):
    model = ARIMA(data["value"], order=(p, d, q))
    model_fit = model.fit()
    model_summary = model_fit.summary()
    print(model_summary)
    
def main():
    data = pd.read_csv('data/dataset.txt', sep=',', skipinitialspace=True)
    data = EDA(data)
    ARIMA_EDA(data, lags=50)
    auto_arima_model(data)


    ARIMA_model(data, p=2, d=1, q=3)
    ARIMA_model(data, p=2, d=1, q=3)

if __name__ == "__main__":
    main()
