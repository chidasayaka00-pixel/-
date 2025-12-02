import pandas as pd
import numpy as np

def compute_features_by_pair(df):

    feature_list = []

    for (client, pair), dfc in df.groupby(['client_name', 'currency_pair_iso']):

        dfc = dfc.sort_values('trade_date')
        flow = dfc['net_volume_traded_jpm_usd'].values

        # skip very small histories
        if len(flow) < 5:
            continue

        sign = np.sign(flow)
        sign_changes = np.sum(np.diff(sign) != 0)

        f = {
            'client_name': client,
            'currency_pair': pair,

            # --- directional behavior ---
            'lag1_autocorr': pd.Series(sign).autocorr(lag=1),
            'lag3_autocorr': pd.Series(sign).autocorr(lag=3),
            'switch_rate': sign_changes / (len(flow) - 1),
            'avg_run_length': len(flow) / (sign_changes + 1),

            # --- product mix (CTA = forward / NDF heavy) ---
            'pct_spot': np.mean(dfc['product_type'] == 'Spot'),
            'pct_forward': np.mean(dfc['product_type'] == 'Forward'),
            'pct_ndf': np.mean(dfc['product_type'] == 'NDF'),
            'pct_swap': np.mean(dfc['product_type'] == 'Swap'),

            # --- flow characteristics ---
            'flow_volatility': np.std(flow),
            'num_trades': len(flow)
        }

        feature_list.append(f)

    return pd.DataFrame(feature_list)


def score_cta_behavior(df_feat):

    df = df_feat.copy()

    # normalize volatility (lower vol = more CTA-like)
    vol_rank = df['flow_volatility'].rank(pct=True)
    df['vol_score'] = 1 - vol_rank

    # directional persistence
    df['trend_score'] = df['lag1_autocorr'].clip(-1, 1)
    df['stability_score'] = (1 - df['switch_rate']).clip(0, 1)

    # run-length normalization
    df['run_score'] = df['avg_run_length'] / df['avg_run_length'].max()

    # product mix score (forward/NDF good, spot bad)
    df['product_score_raw'] = (
        df['pct_forward'] * 1.0 +
        df['pct_ndf'] * 1.0 +
        df['pct_swap'] * 0.3 -
        df['pct_spot'] * 0.8
    )

    # normalize product score to 0–1
    min_p = df['product_score_raw'].min()
    max_p = df['product_score_raw'].max()
    df['product_score'] = (df['product_score_raw'] - min_p) / (max_p - min_p + 1e-9)

    df['CTA_pair_score'] = (
        0.35 * df['trend_score'] +
        0.25 * df['stability_score'] +
        0.15 * df['run_score'] +
        0.15 * df['product_score'] +
        0.10 * df['vol_score']
    )

    df['CTA_pair_flag'] = df['CTA_pair_score'] >= 0.60

    return df



def aggregate_client_cta(df_scored):

    summary = (
        df_scored.groupby('client_name')['CTA_pair_flag']
        .mean()
        .rename('cta_ratio')
        .to_frame()
    )

    summary['CTA_client_flag'] = summary['cta_ratio'] >= 0.50

    return summary.sort_values('cta_ratio', ascending=False)



df_feat = compute_features_by_pair(df)
df_scored = score_cta_behavior(df_feat)
cta_clients = aggregate_client_cta(df_scored)

print(cta_clients.head(20))



\\\\\\\\\\\API:

def convert_pair_to_api_format(pair):
    return pair.replace('/', '-')

convert_pair_to_api_format("USD/JPY") → "USD-JPY"
convert_pair_to_api_format("USD/NZD") → "USD-NZD"

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


price_dict = {
    "USD/JPY":   df_price_for_usdjpy,
    "USD/NZD":   df_price_for_usdnzd,
    "EUR/USD":   df_price_for_eurusd,
    ...
}


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

C/////////omute Trend beta:
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

Updated regression code:
from sklearn.linear_model import LinearRegression

def compute_trend_betas_scaled(df_flow, price_dict, lags=[1,3,5,10]):

    df_flow['trade_date'] = pd.to_datetime(df_flow['trade_date'])
    beta_rows = []

    for (client, pair), dfc in df_flow.groupby(['client_name','currency_pair_iso']):

        if pair not in price_dict:
            continue

        df_price = price_dict[pair]
        merged = dfc.merge(df_price, left_on='trade_date', right_index=True, how='inner')

        if len(merged) < 10:
            continue

        beta_dict = {
            'client_name': client,
            'currency_pair': pair
        }

        # scale flow to "millions"
        merged['flow_mn'] = merged['net_volume_traded_jpm_usd'] / 1e6

        for lag in lags:
            col = f'ret_lag{lag}'
            if col not in merged.columns:
                beta_dict[f'beta_lag{lag}'] = np.nan
                continue

            # convert returns to bps (1bp = 0.0001)
            merged[f'{col}_bps'] = merged[col] * 10000

            valid = merged.dropna(subset=[f'{col}_bps','flow_mn'])
            if len(valid) < 10:
                beta_dict[f'beta_lag{lag}'] = np.nan
                continue

            X = valid[[f'{col}_bps']].values
            y = valid['flow_mn'].values

            model = LinearRegression().fit(X, y)
            beta_dict[f'beta_lag{lag}'] = model.coef_[0]   # flow_mn per bp

        beta_rows.append(beta_dict)

    return pd.DataFrame(beta_rows)


\\\\\some fix:
# ensure column names are strings
df_betas.columns = df_betas.columns.astype(str)

# extract beta columns safely
beta_cols = [c for c in df_betas.columns if c.startswith("beta_")]

# scale those columns to millions
if len(beta_cols) > 0:
    df_betas[beta_cols] = df_betas[beta_cols].astype(float) / 1e6
else:
    print("No beta columns found!")


