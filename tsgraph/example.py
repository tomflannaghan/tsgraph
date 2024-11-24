import numpy as np
import pandas as pd
from matplotlib import pyplot
from pyg_base import dictable

from tsgraph.nodes.align import pack, get_col
from tsgraph.nodes.core import df_node, output_node
from tsgraph.nodes.maths import div, ewma
from tsgraph.nodes.utils import stitch, diff, lag


def fake_spot_series(std, initial_price):
    dates = pd.bdate_range(pd.Timestamp('1999-01-01'), pd.Timestamp('2020-01-01'), freq='B')
    prices = (1 + np.random.normal(scale=std, size=len(dates))).cumprod() * initial_price
    return pd.Series(prices, index=dates)


def fake_price_series(fake_spot, contract_date, initial_spread, spread_std, drift):
    spot = fake_spot.loc[contract_date - pd.offsets.BDay(1000):contract_date]
    spread = np.random.normal(loc=drift * initial_spread, scale=spread_std * initial_spread, size=len(spot)).cumsum()
    return spot + spread[::-1]


markets = dictable(
    market=['WTI', 'ES', 'NG', 'DAX'],
    initial_price=[100, 500, 20, 1],
    initial_spread=[5, 5, 2, 1],
    std=0.01,
    drift=0.002,
    spread_std=0.05,
)
markets = markets(spot=fake_spot_series)
contract_data = dictable(
    contract_date=list(pd.date_range(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-01-01'), freq='MS')))
contract_data = contract_data.join(markets)
contract_data = contract_data(price=lambda spot, contract_date, initial_spread, spread_std, drift: df_node(
    fake_price_series(spot, contract_date, initial_spread, spread_std, drift)))
contract_data = contract_data(roll_date=lambda contract_date: contract_date - pd.offsets.BDay(10))
# fake data end, remove the columns used for making the fake data.
contract_data = contract_data[['market', 'price', 'roll_date', 'contract_date']]

contract_data = contract_data(price_data=lambda price, contract_date: pack({
    'price_unadj': price, 'price_delta': diff(price, 1), 'return': div(diff(price, 1), lag(price, 1)),
    'contract_date': contract_date
}))

market_data = contract_data.listby('market')
market_data = market_data(price_data=lambda price_data, roll_date: output_node(stitch(price_data, roll_date)))

market_data = market_data(signal=lambda price_data: ewma(get_col(price_data, 'return'), 10))

df = market_data.inc(market='WTI')[0].price_data.calc()

backadjust = (1 + df['return']).cumprod()
backadjust *= df['price_unadj'].iloc[-1] / backadjust.iloc[-1]
pyplot.plot(df['price_unadj'], label='price_unadj')
pyplot.plot(backadjust, label='backadjust')
pyplot.legend()
pyplot.show()