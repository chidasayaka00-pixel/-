# # --- STEP 1: Build Daily Flow Matrix (client-level) ---

df_daily_client = (
    df.groupby(['trade_date', 'client_name'])['flow']
      .sum()
      .unstack(fill_value=0)
      .sort_index()
)

df_daily_client.head()




import pandas as pd
import numpy as np

def quick_features(df, client):
    """Extract quick behavior features for a given client_name."""
    
    # filter this client's trades only
    dfc = df[df['client_name'] == client].copy()
    dfc = dfc.sort_values('trade_date')
    
    flow = dfc['flow'].values
    sign = np.sign(flow)
    
    features = {}
    
    # --- 1. Persistence / Direction Behavior ---
    features['lag1_autocorr'] = pd.Series(sign).autocorr(lag=1)
    features['lag3_autocorr'] = pd.Series(sign).autocorr(lag=3)
    
    # switching frequency: how often direction changes
    switch = np.sum(np.diff(sign) != 0)
    features['switch_rate'] = switch / (len(flow) - 1)
    
    # run length = average continuous direction length
    features['avg_run_length'] = len(flow) / (switch + 1)
    
    # volatility of flow size
    features['flow_volatility'] = np.std(flow)
    
    # --- 2. Product Mix (CTA important feature) ---
    features['pct_spot'] = np.mean(dfc['product_type'] == 'spot')
    features['pct_forward'] = np.mean(dfc['product_type'] == 'forward')
    features['pct_swap'] = np.mean(dfc['product_type'] == 'swap')
    features['pct_ndf'] = np.mean(dfc['product_type'] == 'ndf')
    
    return features



all_clients = df['client_name'].unique()

feature_list = []
for client in all_clients:
    f = quick_features(df, client)
    f['client_name'] = client
    feature_list.append(f)

df_features = pd.DataFrame(feature_list).set_index('client_name')
df_features.head()




import numpy as np
import pandas as pd

def compute_cta_score(df_features):
    scores = pd.DataFrame(index=df_features.index)

    # 1. Trend following persistence
    scores['trend_score'] = (
        0.6 * df_features['lag1_autocorr'].clip(lower=-1, upper=1) +
        0.4 * df_features['lag3_autocorr'].clip(lower=-1, upper=1)
    )

    # 2. Direction stability (inverse of switch_rate)
    scores['stability_score'] = (1 - df_features['switch_rate']).clip(0, 1)

    # 3. Run-length (longer = more CTA-like)
    max_run = df_features['avg_run_length'].max()
    scores['run_score'] = df_features['avg_run_length'] / max_run

    # 4. Product mix: CTA uses forwards, not spot
    scores['product_score'] = (
        1.0 * df_features['pct_forward'] +
        0.5 * df_features['pct_swap'] -
        0.8 * df_features['pct_spot']
    )

    # 5. Execution consistency (inverse of flow volatility rank)
    vol_rank = df_features['flow_volatility'].rank(pct=True)
    scores['execution_score'] = 1 - vol_rank  # lower volatility → more CTA-like

    # normalize each component to [0,1]
    for col in scores.columns:
        col_min, col_max = scores[col].min(), scores[col].max()
        if col_max > col_min:
            scores[col] = (scores[col] - col_min) / (col_max - col_min)

    # final CTA score (weighted)
    scores['CTA_score'] = (
        0.35 * scores['trend_score'] +
        0.25 * scores['stability_score'] +
        0.15 * scores['run_score'] +
        0.15 * scores['product_score'] +
        0.10 * scores['execution_score']
    )

    return scores

def flag_cta(df_features, threshold=0.65):
    scores = compute_cta_score(df_features)
    scores['CTA_flag'] = scores['CTA_score'] >= threshold
    return scores.sort_values('CTA_score', ascending=False)



cta_scores = flag_cta(df_features)
cta_scores.head(20)



