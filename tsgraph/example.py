import numpy as np
import pandas as pd
from pyg_base import dictable

from tsgraph.nodes.align import pack
from tsgraph.nodes.core import df_node
from tsgraph.nodes.utils import stitch, diff


def fake_price_series(contract_month):
    prices = (1 + np.random.normal(scale=0.01, size=250)).cumprod()
    dates = pd.bdate_range(contract_month - pd.offsets.BDay(250), contract_month, freq='B')
    return pd.Series(prices, index=dates[-len(prices):])


markets = dictable(market=['WTI', 'ES', 'NG', 'DAX'])
contract_data = dictable(
    contract_month=list(pd.date_range(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-01-01'), freq='MS')))
contract_data = contract_data.join(markets)
contract_data = contract_data(price=lambda contract_month: df_node(fake_price_series(contract_month)))
contract_data = contract_data(roll_date=lambda contract_month: contract_month - pd.offsets.BDay(10))
contract_data = contract_data(price_data=lambda price, contract_month: pack({
    'price_unadj': price, 'price_delta': diff(price, 1), 'contract_month': contract_month
}))

market_data = contract_data.listby('market')
market_data = market_data(price_data=lambda price_data, roll_date: stitch(price_data, roll_date))

market_data[0].price_data.calc()