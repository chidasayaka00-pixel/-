# # --- STEP 1: Build Daily Flow Matrix (client-level) ---

df_daily_client = (
    df.groupby(['trade_date', 'client_name'])['flow']
      .sum()
      .unstack(fill_value=0)
      .sort_index()
)

df_daily_client.head()
