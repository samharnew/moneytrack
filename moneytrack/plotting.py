from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from .config import Config
from .moneyframe import MoneyFrame
from .moneyframecollection import MoneyFrameCollection
from .utils import assert_type
import pandas as pd

field_names = Config.FieldNames
register_matplotlib_converters()


class MoneyViz:
    class Metric(Enum):
        Balance = 1
        Interest = 2
        Transfers = 3
        InterestRate = 4

    def_labels = {
        Metric.InterestRate: "Interest Rate",
        Metric.Interest: "Interest Payments",
        Metric.Transfers: "Transfer Amount",
        Metric.Balance: "Balance"
    }
    def_units = {
        Metric.InterestRate: "[%]",
        Metric.Interest: "[£]",
        Metric.Transfers: "[£]",
        Metric.Balance: "[£]"
    }
    def_annotate_fun = {
        Metric.InterestRate: lambda x: "{} %".format(round(x, 2)),
        Metric.Interest: lambda x: "£{:,}".format(round(x, 2)),
        Metric.Transfers: lambda x: "£{:,}".format(round(x, 2)),
        Metric.Balance: lambda x: "£{:,}".format(round(x, 2)),
    }

    @classmethod
    def get_figure(cls, ax=None):
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.get_figure()
        return f, ax

    @classmethod
    def get_metric(cls, mf: MoneyFrame, metric: Metric, cumulative: bool = False):

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

        return s

    @classmethod
    def get_normalised_metric(cls, mf: MoneyFrame, mf_norm: MoneyFrame, metric: Metric, cumulative: bool = False):

        y = cls.get_metric(mf, metric=metric, cumulative=cumulative)
        y_norm = cls.get_metric(mf_norm, metric=metric, cumulative=cumulative)
        return ((y/y_norm)*100.0).fillna(0.0)

    @classmethod
    def plot_mf(cls, mf: MoneyFrame, metric: Metric, cumulative: bool = False, ax=None, **kwargs):

        assert_type(mf, MoneyFrame)
        assert_type(metric, cls.Metric)

        f, ax = cls.get_figure(ax=ax)
        s = cls.get_metric(mf, metric, cumulative=cumulative)
        ax.plot(s, **kwargs)

        return f, ax

    @classmethod
    def plot_mfc(cls, mfc: MoneyFrameCollection, metric: Metric, cumulative: bool = False, ax=None, **kwargs):

        for k, mf in mfc.items():
            f, ax = cls.plot_mf(mf, metric, cumulative=cumulative, ax=ax, label=k, **kwargs)
        return f, ax

    @classmethod
    def plot_mfc_normalized(cls, mfc: MoneyFrameCollection, metric: Metric, cumulative: bool = False,
                            stacked: bool = False, ax=None, **kwargs):

        f, ax = cls.get_figure(ax=ax)

        s = {
            k: cls.get_normalised_metric(mf, mfc.sum(), metric=metric, cumulative=cumulative)
            for k, mf in mfc.items()
        }

        if stacked:
            y_pos = [si.clip(lower=0.0) for si in s.values()]
            y_neg = [si.clip(upper=0.0) for si in s.values()]
            plots = ax.stackplot(y_pos[0].index, *y_pos, labels=list(s.keys()), **kwargs)
            colors = [p.get_facecolors()[0] for p in plots]
            ax.stackplot(y_neg[0].index, *y_neg, **kwargs, colors=colors)
            ax.margins(0.0, 0.0)
        else:
            for k, si in s.items():
                ax.plot(si, label=k, **kwargs)

        return f, ax

    @classmethod
    def construct_label(cls, metric: Metric, cumulative: bool = False, normalize: bool = False):

        label = cls.def_labels[metric]
        label = "Cumulative " + label if cumulative else label

        if normalize:
            label = "% of Total " + label
        else:
            label = label + " " + cls.def_units[metric]
        return label

    @classmethod
    def plot(cls, mf: Union[MoneyFrame, MoneyFrameCollection], metric: Metric, cumulative: bool = False,
             normalize: bool = False, ax = None, stacked: bool = False, **kwargs):

        if isinstance(mf, MoneyFrame):
            if normalize:
                normalize = False
            f, ax = cls.plot_mf(mf, metric=metric, cumulative=cumulative, ax=ax, **kwargs)
        elif isinstance(mf, MoneyFrameCollection) and not normalize:
            f, ax = cls.plot_mfc(mf, metric=metric, cumulative=cumulative, ax=ax, **kwargs)
        elif isinstance(mf, MoneyFrameCollection) and normalize:
            f, ax = cls.plot_mfc_normalized(mf, metric=metric, cumulative=cumulative, ax=ax, stacked=stacked, **kwargs)
        else:
            raise TypeError("Can only plot MoneyFrame or MoneyFrameCollection objects")

        ax.set_xlabel("Date")
        ax.set_ylabel(cls.construct_label(metric, cumulative, normalize))

        return f, ax

    @classmethod
    def calculate_bar_metric(cls, mf: MoneyFrame, metric: Metric, normalize: bool = False):

        if metric == MoneyViz.Metric.Balance:
            s = mf.get_daily_balance().iloc[-1]
        elif metric == MoneyViz.Metric.Interest:
            s = mf.get_daily_interest().sum()
        elif metric == MoneyViz.Metric.Transfers:
            s = mf.get_daily_transfers().sum()
        elif metric == MoneyViz.Metric.InterestRate:
            s = mf.calc_avg_interest_rate()
        else:
            raise AttributeError("Not implemented plotting for metric={}".format(metric))

        return s

    @classmethod
    def calculate_bar_metrics(cls, mfc: MoneyFrameCollection, metric: Metric, normalize: bool = False) -> pd.Series:

        results = {k: cls.calculate_bar_metric(mf, metric, normalize) for k, mf in mfc.items()
                   if not mf.is_zombie()}
        results = pd.Series(results, name="metric").sort_values(ascending=False)

        if metric in (MoneyViz.Metric.Balance,):
            results = results[results.values != 0.0]

        if normalize:
            if metric in (MoneyViz.Metric.InterestRate, ):
                raise ValueError("Should not normalise the metric 'InterestRate'")
            results = (results/results.sum())*100.0

        return results

    @staticmethod
    def label_bars(bars, ax, value_to_label_func=None, rotation=35):

        for rect in bars:
            y = rect.get_height()
            label = str(y) if value_to_label_func is None else value_to_label_func(y)
            x_center = rect.get_x() + rect.get_width() / 2
            va = 'bottom' if y >= 0 else 'top'
            rot = rotation if y >= 0 else -rotation
            xy_text = (0, 3) if y >= 0 else (0, -3)
            ax.annotate(label, xy=(x_center, y), xytext=xy_text, rotation=rot,
                        textcoords="offset points", ha='left', va=va, fontweight='bold', )

    @classmethod
    def bar(cls, mfc: Union[MoneyFrameCollection, MoneyFrame], metric: Metric, ax=None, annotate=True, normalize=False, **kwargs):

        if isinstance(mfc, MoneyFrame):
            mfc = mfc.as_collection()

        f, ax = cls.get_figure(ax=ax)

        results = cls.calculate_bar_metrics(mfc, metric, normalize)

        x = results.index.values.astype(str)
        y = results

        bars = ax.bar(x, y, **kwargs)
        ax.set_ylabel(cls.construct_label(metric, normalize=normalize))
        ax.axhline(0.0, color='black', ls='-', lw=1)

        if annotate:
            if normalize:
                MoneyViz.label_bars(bars, ax, lambda x: "{:.2f}%".format(x))
            else:
                MoneyViz.label_bars(bars, ax, cls.def_annotate_fun[metric])
            ax.margins(y=0.1)
        return f, ax

    @classmethod
    def pie(cls, mfc: Union[MoneyFrameCollection, MoneyFrame], metric: Metric, ax=None, explode: float = 0.0):

        if isinstance(mfc, MoneyFrame):
            mfc = mfc.as_collection()

        f, ax = cls.get_figure(ax=ax)

        results = cls.calculate_bar_metrics(mfc, metric)

        autopct = '%1.2f%%'
        ax.pie(results.values, explode=[explode]*len(results), labels=results.index, autopct=autopct, shadow=False,
               startangle=180)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        return f, ax
