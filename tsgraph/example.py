import numpy as np
import pandas as pd
from pyg_base import dictable

from tsgraph.node import df_node

def fake_price_series(contract_month):
    prices = (1 + np.random.normal(scale=0.01, size=250)).cumprod()
    dates = pd.bdate_range(contract_month - pd.offsets.BDay(250), contract_month, freq='B')
    return pd.Series(prices, index=dates[-len(prices):])


markets = dictable(markets=['WTI', 'ES', 'NG', 'DAX'])
contract_data = dictable(contract_month=list(pd.date_range(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-01-01'), freq='MS')))
contract_data = contract_data.join(markets)
contract_data = contract_data(price=lambda contract_month: df_node(fake_price_series(contract_month)))
contract_data = contract_data(roll_date=lambda contract_month: contract_month - pd.offsets.BDay(10))
