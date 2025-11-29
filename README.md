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
