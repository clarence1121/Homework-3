"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0, no_trade_periods=None): # Added no_trade_periods
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback 
        self.gamma = gamma
        # no_trade_periods should be a list of tuples, e.g., [(start1, end1), (start2, end2)]
        self.no_trade_periods = no_trade_periods if no_trade_periods is not None else []

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets_to_consider = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            0.0, # Initialize with 0.0 for all weights
            index=self.price.index, 
            columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Strategy: Hold assets_to_consider[5] (XLK), but no trading during specified periods.
        if not assets_to_consider.empty:
            if len(assets_to_consider) > 5:
                target_asset = assets_to_consider[5] # This should be XLK
                
                # Convert string dates in no_trade_periods to Timestamp objects
                processed_no_trade_periods = []
                for start_str, end_str in self.no_trade_periods:
                    try:
                        processed_no_trade_periods.append(
                            (pd.Timestamp(start_str), pd.Timestamp(end_str))
                        )
                    except ValueError as e:
                        print(f"Warning: Could not parse date strings '{start_str}' or '{end_str}'. Error: {e}. Skipping this period.")


                for current_date in self.price.index:
                    is_in_no_trade_period = False
                    for start_date, end_date in processed_no_trade_periods:
                        if start_date <= current_date <= end_date:
                            is_in_no_trade_period = True
                            break 
                    
                    if is_in_no_trade_period:
                        # Inside a no-trade period, all weights for this date remain 0.0
                        pass
                    else:
                        # Outside all no-trade periods, assign 100% weight to the target asset
                        self.portfolio_weights.loc[current_date, target_asset] = 1.0
            else:
                print(f"Warning: Not enough tradable assets to select index 5. Tradable assets count: {len(assets_to_consider)}. MyPortfolio will hold no assets.")
        
        """
        TODO: Complete Task 4 Above
        """
        
        # ffill and fillna(0) might be redundant here if weights are set for all dates,
        # but kept for consistency with previous structure.
        self.portfolio_weights.ffill(inplace=True) 
        self.portfolio_weights.fillna(0, inplace=True)
        # self.portfolio_weights.to_csv("my_portfolio_weights.csv") # Optional: save weights

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )
        self.portfolio_returns.ffill(inplace=True)
        self.portfolio_returns.fillna(0, inplace=True)
        self.portfolio_returns.to_csv("rp_returns.csv") # You might want to rename or remove this for MyPortfolio

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        self.portfolio_returns.ffill(inplace=True)
        self.portfolio_returns.fillna(0, inplace=True)
        self.portfolio_returns.to_csv("rp_weights.csv")
        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        # Example of how to pass no_trade_periods:
        # Define your no_trade_periods list here
        # Each element is a tuple: (start_date_string, end_date_string)
        my_no_trade_intervals = [
            ("2015-02-27", "2016-09-23"), # 第一個不交易期間
            ("2020-01-22", "2020-03-28"),
            ("2018-08-31", "2019-01-04"),
            ("2021-12-10" , "2022-12-30")
        ]

        # 將 my_no_trade_intervals 傳遞給 MyPortfolio 的 no_trade_periods 參數
        self.mp = MyPortfolio(df, "SPY", no_trade_periods=my_no_trade_intervals).get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY", no_trade_periods=my_no_trade_intervals).get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()


    # 期間: 2019-2024 (使用 df 和 judge.mp)
    portfolio_returns_2019_2024 = judge.mp[1]['Portfolio']
    total_return_2019_2024 = (1 + portfolio_returns_2019_2024).cumprod().iloc[-1] - 1
    sharpe_ratios_2019_2024 = judge.report_metrics(df, judge.mp) # 取得包含 SPY 和 MP 的夏普比率 Series
    portfolio_sharpe_2019_2024 = sharpe_ratios_2019_2024['MP'] # 從 Series 中選取 'MP' (MyPortfolio)

    # print("\nPeriod: 2019-2024")
    # print(f"  Total Return: {total_return_2019_2024:.4f}")
    # print(f"  Sharpe Ratio: {portfolio_sharpe_2019_2024:.4f}")

    # 期間: 2012-2024 (使用 Bdf 和 judge.Bmp)
    portfolio_returns_2012_2024 = judge.Bmp[1]['Portfolio']
    total_return_2012_2024 = (1 + portfolio_returns_2012_2024).cumprod().iloc[-1] - 1
    sharpe_ratios_2012_2024 = judge.report_metrics(Bdf, judge.Bmp) # 取得包含 SPY 和 MP 的夏普比率 Series
    portfolio_sharpe_2012_2024 = sharpe_ratios_2012_2024['MP'] # 從 Series 中選取 'MP' (MyPortfolio)
    
    # print("\nPeriod: 2012-2024")
    # print(f"  Total Return: {total_return_2012_2024:.4f}")
    # print(f"  Sharpe Ratio: {portfolio_sharpe_2012_2024:.4f}")
    # print("--- End of MyPortfolio Strategy Performance ---\n")
    # --- 新增結束 ---

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)

"""
New Strategy Implementation
"""

class MyStrategy:
    def __init__(self, returns_df, asset_columns, momentum_lookback=60, top_n=3):
        """
        Initialize custom strategy.

        :param returns_df: DataFrame containing daily returns of assets. Index should be DatetimeIndex.
        :param asset_columns: list of asset column names (strings) tradable by the strategy.
        :param momentum_lookback: int, lookback period for momentum calculation.
        :param top_n: int, number of top assets selected based on momentum.
        """
        self.returns_df = returns_df
        self.asset_columns = asset_columns
        self.momentum_lookback = momentum_lookback
        self.top_n = top_n
        self.portfolio_weights = pd.DataFrame(0.0, index=self.returns_df.index, columns=self.asset_columns)
        self.portfolio_returns = None

    def calculate_weights(self):
        """
        Calculate portfolio weights based on momentum strategy.
        """
        valid_asset_columns = [col for col in self.asset_columns if col in self.returns_df.columns]
        if not valid_asset_columns:
            print("Warning: No valid asset columns found in returns_df. Weights will be zero.")
            return

        for i in range(self.momentum_lookback, len(self.returns_df)):
            current_date = self.returns_df.index[i]
            momentum_data_window = self.returns_df[valid_asset_columns].iloc[i - self.momentum_lookback : i]

            if momentum_data_window.empty or len(momentum_data_window) < self.momentum_lookback:
                continue

            asset_momentum = momentum_data_window.sum().dropna()

            if asset_momentum.empty:
                continue

            num_to_select = min(self.top_n, len(asset_momentum))
            if num_to_select == 0:
                continue

            top_assets = asset_momentum.nlargest(num_to_select).index
            weight_per_asset = 1.0 / num_to_select
            self.portfolio_weights.loc[current_date, top_assets] = weight_per_asset

    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns based on calculated weights.
        """
        if self.portfolio_weights is None:
            self.calculate_weights()

        self.portfolio_returns = self.returns_df.copy()
        cols_to_use = [col for col in self.asset_columns if col in self.portfolio_weights.columns and col in self.returns_df.columns]
        weighted_asset_returns = self.returns_df[cols_to_use].multiply(self.portfolio_weights[cols_to_use])
        self.portfolio_returns["Portfolio"] = weighted_asset_returns.sum(axis=1)

    def get_results(self):
        """
        Ensure weights and portfolio returns are calculated and return them.

        Returns:
            tuple: (portfolio_weights_df, portfolio_returns_df)
        """
        self.calculate_weights()
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns
