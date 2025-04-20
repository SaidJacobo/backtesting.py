from backtesting.test import USDJPY
from backtesting.test import GBPNZD
from backtesting.test import NZDUSD
from backtesting.test import EURUSD
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
pd.set_option('display.max_columns', None)

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy(size=100_000)

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell(size=100_000)



if __name__ == '__main__':

    # print('GBPNZD', GBPNZD.head())
    # print('NZDUSD', NZDUSD.head())

    # # GBPNZD['ConversionRate'] = NZDUSD['Open']
    NZDUSD = NZDUSD[['Open']].rename(columns={'Open':'ConversionRate'})

    GBPNZD = pd.merge(
        GBPNZD,
        NZDUSD,
        how='inner',
        left_index=True,
        right_index=True
    )

    # print('GBPNZD', GBPNZD.head())

    # USDJPY['ConversionRate'] = 1 / USDJPY['Open']

    bt = Backtest(GBPNZD, SmaCross, cash=100_000, commission=(0, 0.0001), margin=1/30)
    stats = bt.run()
    print(stats)
    print(stats._trades)
    print(stats._equity_curve.tail(50))



