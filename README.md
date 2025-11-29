def convert_pair_to_api_format(pair):
    return pair.replace('/', '-')

unique_pairs = df['currency_pair_iso'].unique()

import tsfns
import datetime
import pandas as pd

def fetch_all_fx_timeseries(df, start_date, end_date):
    """
    Fetch price timeseries for ALL currency pairs in your flow dataset.
    Returns dict: {'USD/JPY': df_price, 'USD/NZD': df_price, ...}
    """

    price_dict = {}

    for pair in df['currency_pair_iso'].unique():
        api_pair = convert_pair_to_api_format(pair)

        try:
            t = tsfns.TimeSeries(api_pair, 
                                 StartDate=start_date, 
                                 EndDate=end_date)
            raw = t.Pairs()  # [[date, price], ...]

            # convert raw data to df
            df_price = pd.DataFrame(raw, columns=['date', pair])
            df_price['date'] = pd.to_datetime(df_price['date'])
            df_price = df_price.set_index('date').sort_index()

            price_dict[pair] = df_price

        except Exception as e:
            print(f"Failed to fetch {pair}: {e}")

    return price_dict



price_dict = fetch_all_fx_timeseries(df, 
                                     start_date=datetime.date(2025,1,1),
                                     end_date=datetime.date(2025,11,1))


def compute_returns(price_dict, lags=[1,3,5,10]):
    updated = {}

    for pair, df_price in price_dict.items():
        dfp = df_price.copy()
        dfp['ret'] = dfp[pair].pct_change()

        for lag in lags:
            dfp[f'ret_lag{lag}'] = dfp['ret'].shift(lag)

        updated[pair] = dfp

    return updated

price_dict = compute_returns(price_dict)

from sklearn.linear_model import LinearRegression
import numpy as np

def compute_trend_betas(df_flow, price_dict, lags=[1,3,5,10]):
    """
    df_flow columns must include:
    - trade_date
    - client_name
    - currency_pair_iso
    - net_volume_traded_jpm_usd
    """

    df_flow['trade_date'] = pd.to_datetime(df_flow['trade_date'])

    beta_rows = []

    for (client, pair), dfc in df_flow.groupby(['client_name','currency_pair_iso']):

        # skip if no price data
        if pair not in price_dict:
            continue

        df_price = price_dict[pair]

        # merge flow + price returns
        merged = dfc.merge(df_price, left_on='trade_date', right_index=True, how='inner')

        if len(merged) < 10:
            continue

        beta_dict = {
            'client_name': client,
            'currency_pair': pair
        }

        for lag in lags:
            col = f'ret_lag{lag}'
            if col not in merged.columns:
                beta_dict[f'beta_lag{lag}'] = np.nan
                continue

            valid = merged.dropna(subset=[col, 'net_volume_traded_jpm_usd'])
            if len(valid) < 10:
                beta_dict[f'beta_lag{lag}'] = np.nan
                continue

            X = valid[[col]].values.reshape(-1,1)
            y = valid['net_volume_traded_jpm_usd'].values

            model = LinearRegression().fit(X, y)
            beta_dict[f'beta_lag{lag}'] = model.coef_[0]

        beta_rows.append(beta_dict)

    return pd.DataFrame(beta_rows)


df_betas = compute_trend_betas(df, price_dict)

