from datetime import datetime
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from .config import Config
from .core import MoneyData
from .moneyframe import MoneyFrame
from .moneyframecollection import MoneyFrameCollection
from .utils import assert_type

field_names = Config.FieldNames
register_matplotlib_converters()


class MoneyViz:
    class Metric(Enum):
        Balance = 1
        Interest = 2
        Transfers = 3
        InterestRate = 4

    def_labels = {
        Metric.InterestRate: "Interest Rate [%]",
        Metric.Interest: "Interest Payments [£]",
        Metric.Transfers: "Transfer Amount [£]",
        Metric.Balance: "Balance [£]"
    }

    @classmethod
    def get_figure(cls, ax=None):
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.get_figure()
        return f, ax

    @classmethod
    def plot_mf(cls, mf: MoneyFrame, metric: Metric, cumulative: bool = False, ax=None, **kwargs):

        assert_type(mf, MoneyFrame)
        assert_type(metric, cls.Metric)

        f, ax = cls.get_figure(ax=ax)

        if metric == MoneyViz.Metric.Balance:
            s = mf.get_daily_balance()
            assert cumulative is False, "Plotting cumulative balance makes NO sense. Rethink"
        elif metric == MoneyViz.Metric.Interest:
            s = mf.get_daily_interest()
            if cumulative:
                s = s.cumsum()
        elif metric == MoneyViz.Metric.Transfers:
            s = mf.get_daily_transfers()
            if cumulative:
                s = s.cumsum()
        elif metric == MoneyViz.Metric.InterestRate:
            s = mf.get_daily_interest_rate()
            if cumulative:
                s = mf.get_cumulative_interest_rate()
        else:
            raise AttributeError("Not implemented plotting for metric={}".format(metric))

        ax.plot(s, **kwargs)

        return f, ax

    @classmethod
    def plot_mfc(cls, mfc: MoneyFrameCollection, metric: Metric, cumulative: bool = False, ax=None, **kwargs):

        for k, mf in mfc.items():
            f, ax = cls.plot_mf(mf, metric, cumulative=cumulative, ax=ax, label=k, **kwargs)
        return f, ax

    @classmethod
    def plot(cls, mf: Union[MoneyFrame, MoneyFrameCollection], metric: Metric, cumulative: bool = False,
             ax=None, **kwargs):

        if isinstance(mf, MoneyFrame):
            f, ax = cls.plot_mf(mf, metric=metric, cumulative=cumulative, ax=ax, **kwargs)
        elif isinstance(mf, MoneyFrameCollection):
            f, ax = cls.plot_mfc(mf, metric=metric, cumulative=cumulative, ax=ax, **kwargs)
        else:
            raise TypeError("Can only plot MoneyFrame or MoneyFrameCollection objects")

        y_label = cls.def_labels[metric]
        if cumulative:
            y_label = "Cumulative " + y_label

        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)

        return f, ax

    @classmethod
    def bar(self, mfc: MoneyFrameCollection, metric: Metric, cumulative: bool = False,
            ax=None, **kwargs):
        pass


class MoneyPlot:
    class Metric(Enum):
        Balance = 1
        Interest = 2
        Transfers = 3
        InterestRate = 4

    def __init__(self, money_data: MoneyData):
        self.money_data = money_data

    @staticmethod
    def get_figure(ax=None):
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.get_figure()
        return f, ax

    def plot_timeseries(
            self,
            metric: Metric = Metric.Balance,
            ax=None,
            agg: bool = True,
            start_date: Union[str, datetime, None] = None,
            end_date: Union[str, datetime, None] = None,
            cumulative: bool = False,
            filters=None,
            **plt_kwargs
    ):

        df = self.money_data.filter_accounts(filters).groupby_accounts(agg).to_df(
            inc_interest_rate=(not cumulative and metric == MoneyPlot.Metric.InterestRate),
            inc_cum_interest_rate=(cumulative and metric == MoneyPlot.Metric.InterestRate),
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
        )
        agg_col = agg
        if agg is True:
            agg_col = "ALL"
        if agg is False:
            agg_col = field_names.ACCOUNT_KEY

        f, ax = MoneyPlot.get_figure(ax=ax)
        ax.set_xlabel("Date")

        def cum_func(x):
            return np.cumsum(x)

        if metric == MoneyPlot.Metric.Balance:
            ax.set_ylabel("Balance [£]")
            col_name = field_names.BALANCE
            title = "Account Balance"
            assert cumulative is False, "Plotting the cumulative balance makes NO sense. Rethink"
        elif metric == MoneyPlot.Metric.Interest:
            ax.set_ylabel("Interest [£]")
            col_name = field_names.INTEREST
            title = "Interest Payments"
        elif metric == MoneyPlot.Metric.Transfers:
            ax.set_ylabel("Transfer Amount [£]")
            col_name = field_names.TRANSFER
            title = "Account Transfers"
        elif metric == MoneyPlot.Metric.InterestRate:
            ax.set_ylabel("Interest Rate [%]")
            title = "Interest Rate"
            if cumulative:
                col_name = field_names.CUM_INTEREST_RATE

                def cum_func(x):
                    return x
            else:
                col_name = field_names.INTEREST_RATE
        else:
            raise AttributeError("Not implemented plotting for metric={}".format(metric))

        if cumulative:
            title = "Cumulative " + title

        if filters is None:
            title += " - All Accounts"
        else:
            title += " - " + ",".join(["{}={}".format(k, v) for k, v in filters.items()])

        ax.set_title(title)

        for label, df_acc in df.groupby(level=agg_col):
            x = df_acc[col_name].index.get_level_values(field_names.DATE)
            if cumulative:
                ax.plot(x, cum_func(df_acc[col_name].values), label=label, **plt_kwargs)
            else:
                ax.plot(x, df_acc[col_name], label=label, **plt_kwargs)

        f.autofmt_xdate()
        return f, ax

    def plot_period_breakdown(
            self,
            metric: Metric,
            start_date: str,
            end_date: str,
            ax=None,
            agg: bool = False,
            filters=None,
            plot_average: bool = True
    ):

        if metric == MoneyPlot.Metric.InterestRate:
            df = self.money_data.filter_accounts(filters).groupby_accounts(agg).avg_interest_rates(
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                as_prcnt=True,
                as_ayr=True,
                as_df=True,
            )
            col = field_names.INTEREST_RATE
        if metric == MoneyPlot.Metric.Interest:
            df = self.money_data.filter_accounts(filters).groupby_accounts(agg).to_df(
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                as_prcnt=True,
                as_ayr=True
            )
            df = df.groupby(level=agg).sum()
            col = field_names.INTEREST

        f, ax = MoneyPlot.get_figure(ax=ax)
        ax.set_xlabel(df.index.name)
        ax.set_ylabel("Interest Rate % [AYR]")

        bars = ax.bar(df.index.values.astype(str), df[col])

        min_y, max_y = df[col].min(), df[col].max()
        range_y = max_y - min_y

        if plot_average and col == field_names.INTEREST_RATE and agg is not True:
            rate = self.money_data.filter_accounts(filters=filters).groupby_accounts(True).avg_interest_rates(
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                as_prcnt=True, as_ayr=True
            )["ALL"]

            ax.axhline(rate, color='black', ls=':', lw=1.5)
            x_max = ax.get_xlim()[1]
            ax.annotate('{}%'.format(round(rate, 2)),
                        xy=(x_max, rate),
                        xytext=(3, 0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='left', va='center', fontweight='bold')

        ax.set_title("Average interest rates from {} to {} inclusive".format(start_date, end_date))
        ax.set_ylim([min_y - range_y * 0.15, max_y + range_y * 0.15])
        ax.axhline(0.0, color='black', ls='-', lw=1)

        def label_bars(bars, ax):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for rect in bars:
                y = round(rect.get_height(), 1)
                x = rect.get_x() + rect.get_width() / 2
                va = 'bottom' if y >= 0 else 'top'
                xy_text = (0, 3) if y >= 0 else (0, -3)
                ax.annotate('{}%'.format(y), xy=(x, y), xytext=xy_text,
                            textcoords="offset points", ha='center', va=va, fontweight='bold')

        label_bars(bars, ax)

        return f, ax
