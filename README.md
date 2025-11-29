import pandas as pd
import numpy as np
def compute_features_by_pair(df):

    feature_list = []

    # group by client_name + currency pair
    for (client, pair), dfc in df.groupby(['client_name', 'currency_pair_iso']):
        dfc = dfc.sort_values('trade_date')

        flow = dfc['flow'].values
        if len(flow) < 5:   # skip too-short series
            continue

        sign = np.sign(flow)
        sign_changes = np.sum(np.diff(sign) != 0)

        f = {
            'client_name': client,
            'currency_pair': pair,
            # --- Directional behavior ---
            'lag1_autocorr': pd.Series(sign).autocorr(lag=1),
            'lag3_autocorr': pd.Series(sign).autocorr(lag=3),
            'switch_rate': sign_changes / (len(flow) - 1),
            'avg_run_length': len(flow) / (sign_changes + 1),

            # --- Product mix ---
            'pct_spot': np.mean(dfc['product_type'] == 'spot'),
            'pct_forward': np.mean(dfc['product_type'] == 'forward'),
            'pct_swap': np.mean(dfc['product_type'] == 'swap'),
            'pct_ndf': np.mean(dfc['product_type'] == 'ndf'),

            # --- Flow characteristics ---
            'flow_volatility': np.std(flow),
            'num_trades': len(flow)
        }

        feature_list.append(f)

    return pd.DataFrame(feature_list)



def score_cta_behavior(df_feat):

    df = df_feat.copy()

    # normalize volatility (lower = more CTA-like)
    vol_rank = df['flow_volatility'].rank(pct=True)
    df['vol_score'] = 1 - vol_rank

    # score components
    df['trend_score'] = df['lag1_autocorr'].clip(lower=-1, upper=1)
    df['stability_score'] = (1 - df['switch_rate']).clip(0, 1)
    df['run_score'] = df['avg_run_length'] / df['avg_run_length'].max()

    # higher forward usage = more CTA-like
    df['product_score'] = (
        df['pct_forward'] * 1.0 +
        df['pct_swap'] * 0.3 -
        df['pct_spot'] * 0.8
    )

    # normalize product_score
    df['product_score'] = (df['product_score'] - df['product_score'].min()) / \
                          (df['product_score'].max() - df['product_score'].min() + 1e-9)

    # final CTA score 0–1
    df['CTA_pair_score'] = (
        0.35 * df['trend_score'] +
        0.25 * df['stability_score'] +
        0.15 * df['run_score'] +
        0.15 * df['product_score'] +
        0.10 * df['vol_score']
    )

    # final flag
    df['CTA_pair_flag'] = df['CTA_pair_score'] >= 0.6

    return df



def aggregate_client_cta(df):
    """
    Input: df = output from score_cta_behavior
    Output: per-client CTA ratio and final flag
    """

    client_summary = (
        df.groupby('client_name')['CTA_pair_flag']
          .mean()
          .rename('cta_ratio')
          .to_frame()
    )

    # threshold: CTA if >50% of currency pairs behave CTA-like
    client_summary['CTA_client_flag'] = client_summary['cta_ratio'] >= 0.5

    return client_summary.sort_values('cta_ratio', ascending=False)


df_feat = compute_features_by_pair(df)
df_scored = score_cta_behavior(df_feat)
cta_clients = aggregate_client_cta(df_scored)

print(cta_clients.head(20))





import yfinance as yf
import pandas as pd

def get_fx_rate(pair, start="2015-01-01"):
    """
    pair format example: 'USD/NZD'
    Yahoo needs 'NZDUSD=X', so invert correctly.
    """

    base, quote = pair.split('/')

    # Yahoo format: QUOTEBASE=X (inverted)
    yahoo_symbol = f"{quote}{base}=X"

    data = yf.download(yahoo_symbol, start=start)

    # The downloaded series is quote/base
    # We want base/quote (USD/NZD)
    data['fx_rate'] = 1 / data['Adj Close']

    return data[['fx_rate']].rename(columns={'fx_rate': pair})
usdnzd = get_fx_rate("USD/NZD", start="2018-01-01")
usdjpy = get_fx_rate("USD/JPY")
