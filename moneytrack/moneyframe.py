import logging
from datetime import datetime, timedelta
from functools import reduce
from typing import Any, Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd

from .config import Config
from .datasets import BalanceUpdates, BalanceTransfers
from .utils import coalesce, assert_type, adr_to_ayr, calc_avg_interest_rate, calc_daily_balances_w_transfers, \
    dates_between, ayr_to_adr
from .exceptions import BalanceExtrapolationError, NoSolutionFoundError

import moneytrack as mt

field_names = Config.FieldNames

log = logging.getLogger("moneyframe")


class MoneyFrame:
    """
    The daily history of a bank account. Contains information about:
        - Any transfers from/to the account each day
        - Any interest payments into the account each day
        - The balance of the account at the end of the day *after* interest payments and transfers
    """

    expected_cols = [
        field_names.INTEREST,
        field_names.TRANSFER,
        field_names.BALANCE,
    ]

    def __init__(self, df: pd.DataFrame):
        """
        Create a MoneyFrame from a pandas DataFrame. The following columns can be included:

            - field_names.BALANCE
            - field_names.DATE
            - field_names.INTEREST
            - field_names.TRANSFER

        Additional columns may be provided, but will be ignored. If the DataFrame is already indexed by date,
        the field_names.DATE field is not needed. The field_names.TRANSFER field is assumed to be zero if not
        provided. If the field_names.INTEREST field is not provided, it is determined by considering the balance
        from the previous day, and any balance transfers into / out of the account.

        :param df: pd.DataFrame
        """

        # Check the DataFrame has all the columns
        assert field_names.BALANCE in df.columns, "DataFrame must have column {}".format(field_names.BALANCE)

        # Make a copy to ensure the original data doesn't get changed.
        df = df.copy()

        # If transfers are not given, assume there are none
        if field_names.TRANSFER not in df.columns:
            df[field_names.TRANSFER] = 0.0

        # If daily interest is not given, calculate from daily balances and transfers
        if field_names.INTEREST not in df.columns:
            df[field_names.INTEREST] = (df[field_names.BALANCE] - df[field_names.TRANSFER] -
                                        df[field_names.BALANCE].shift(1)).fillna(0.0)

        # If the date isn't already the index column, make sure it is
        if field_names.DATE in df.columns:
            df = df.set_index(field_names.DATE)

        # Make sure we only have the columns needed
        df = df[self.expected_cols]

        # Check the data types are legit
        try:
            df = df.astype(float, copy=True)
        except Exception:
            raise ValueError("Some columns of the DataFrame are not compatible with numeric types")

        # Check for missing values
        assert df.isnull().values.any() == 0, "DataFrame contains missing values"

        # Check that the date index is complete
        if len(df) > 0:
            missing_dates = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
            assert len(missing_dates) == 0, "Daily history should contain all dates between the start and end date"

        self.df = df.sort_index(ascending=True)

    def get_slice(self, start_date: Any = None, end_date: Any = None, extrapolate: bool = False):
        """
        Get the MoneyFrame for a specified date range. Optionally an extrapolation of the
        account balance can be made to future / past dates that are not already in the MoneyFrame.

        :param start_date: Any
            First date in the range. If no value is provided, the minimum date will be used.
        :param end_date:
            Last date in the range. If no value is provided, the maximum date will be used.
        :param extrapolate: bool
            If the start_date/end_date is outside the min/max date, should we extrapolate the missing values,
            assuming no interest payments.
        :return: MoneyFrame
        """
        start_date = pd.to_datetime(coalesce(start_date, self.min_date()))
        end_date = pd.to_datetime(coalesce(end_date, self.max_date()))

        if self.min_date() == start_date and self.max_date() == end_date:
            return self

        if (start_date < self.min_date() or end_date > self.max_date()) and extrapolate:
            # Outer join a list of all dates in the range to fill in any gaps
            dates = pd.DataFrame({field_names.DATE: pd.date_range(start_date, end_date)}).set_index(field_names.DATE)
            df = dates.join(self.df, how="outer").sort_index(ascending=True)

            # Extrapolate first/last balances backwards/forwards
            df[field_names.BALANCE] = df[field_names.BALANCE].ffill().bfill()

            # Inner join to select only the range needed. Fill in missing transfer/interest values with 0.
            df = df.loc[start_date:end_date].fillna(0.0)
        else:
            df = self.df.loc[start_date:end_date]

        return MoneyFrame(df)

    def as_collection(self, key=None):
        key = coalesce(key, "ALL ACCOUNTS")
        return mt.MoneyFrameCollection({key: self})

    def to_df(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
              inc_interest_rate=False, inc_cum_interest_rate: bool = False, as_ayr: bool = True,
              as_prcnt: bool = True) -> pd.DataFrame:
        """
        Get pandas DataFrame containing the daily account history. Gives the account balance at the end of the day,
        as well as transfers into/from the account, and any interest payments.

        :param start_date: datetime
            Start the daily summary on this date
        :param end_date: datetime
            End the daily summary on this date
        :param inc_interest_rate: bool
            Include a column containing the interest rate
        :param inc_cum_interest_rate: bool
            Include a column containing the cumulative interest rate
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.DataFrame
            DataFrame containing the daily account summary. It is indexed by date, and has the following columns:
                - field_names.INTEREST
                - field_names.TRANSFER
                - field_names.BALANCE
                - field_names.ACCOUNT_KEY [optional]
        """

        # Check that the inputs are valid
        assert_type(start_date, datetime, optional=True)
        assert_type(end_date, datetime, optional=True)

        if start_date is not None or end_date is not None:
            return self.get_slice(start_date, end_date, extrapolate=True).to_df(
                inc_interest_rate=inc_interest_rate,
                inc_cum_interest_rate=inc_cum_interest_rate,
                as_ayr=as_ayr,
                as_prcnt=as_prcnt,
            )

        df = self.df.copy(deep=False)
        if inc_interest_rate:
            df[field_names.INTEREST_RATE] = self._calc_interest_rate(as_ayr=as_ayr, as_prcnt=as_prcnt)
        if inc_cum_interest_rate:
            df[field_names.CUM_INTEREST_RATE] = self._calc_cum_interest_rate(as_ayr=as_ayr, as_prcnt=as_prcnt)
        return df

    def is_zombie(self, **kwargs):
        """The account is a zombie if it has always had a balance of zero"""
        return (self.get_daily_balance(**kwargs) == 0.0).values.all()

    def min_date(self) -> datetime:
        """Get the minimum date in the daily account history"""
        return self.df.index.min()

    def max_date(self) -> datetime:
        """Get the maximum date in the daily account history"""
        return self.df.index.max()

    def minmax_date(self) -> Tuple[datetime, datetime]:
        """Return a tuple containing the min/max date"""
        return self.min_date(), self.max_date()

    def get_daily_interest(self, **kwargs) -> pd.Series:
        """Amount of interest added to the account each day, indexed by date"""
        return self.to_df(**kwargs)[field_names.INTEREST]

    def get_daily_balance(self, **kwargs) -> pd.Series:
        """Account balance on each day *after* any interest and balance transfers are applied, indexed by date"""
        return self.to_df(**kwargs)[field_names.BALANCE]

    def get_daily_transfers(self, **kwargs) -> pd.Series:
        """Transfers in/out of the account each day, indexed by date."""
        return self.to_df(**kwargs)[field_names.TRANSFER]

    def get_daily_interest_rate(self, **kwargs) -> pd.Series:
        """Series of the daily interest rate, indexed by date."""
        return self.to_df(inc_interest_rate=True, **kwargs)[field_names.INTEREST_RATE]

    def get_cumulative_interest_rate(self, **kwargs) -> pd.Series:
        return self.to_df(inc_cum_interest_rate=True, **kwargs)[field_names.CUM_INTEREST_RATE]

    def get_metric(self, name, **kwargs) -> pd.Series:
        if name == field_names.BALANCE: return self.get_daily_balance(**kwargs)
        if name == field_names.TRANSFER: return self.get_daily_transfers(**kwargs)
        if name == field_names.INTEREST_RATE: return self.get_daily_interest_rate(**kwargs)
        if name == field_names.CUM_INTEREST_RATE: return self.get_cumulative_interest_rate(**kwargs)
        if name == field_names.INTEREST: return self.get_daily_interest(**kwargs)

    @classmethod
    def _interest_rate_conversions(cls, adr: np.array, as_ayr: bool = True, as_prcnt: bool = True) -> np.array:
        """
        Convert a fractional average daily interest rate into (optionally) an average yearly rate (optionally) as
        a percentage.

        :param adr: np.array
            The average daily interest rate as a fraction
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: np.array
        """
        ir = adr_to_ayr(adr) if as_ayr else adr
        ir = ir * 100.0 if as_prcnt else ir
        return ir

    def _calc_interest_rate(self, as_ayr: bool = True, as_prcnt: bool = True) -> pd.Series:
        """
        Calculate the daily interest rate.

        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.Series
            Daily interest rate. The Series is indexed by date
        """
        adr = self.df[field_names.INTEREST] / self.df[field_names.BALANCE].shift(1)
        return self._interest_rate_conversions(adr, as_ayr, as_prcnt)

    @classmethod
    def _calc_avg_interest_rate(cls, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
        """
        Calculate the average interest rate between two dates.

        :param df: pd.DataFrame
            Pandas DataFrame indexed by date with columns field_names.BALANCE, field_names.INTEREST, field_names.TRANSFERS
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: float
            The average interest rate
        """

        # If we have zero days, the interest rate is undefined - return NaN
        if start_date == end_date:
            return np.nan

        # Select the period of interest
        df_sel = df.loc[start_date:end_date]

        # Starting balance before any transfers
        start_bal = df_sel.loc[start_date, field_names.BALANCE] - df_sel.loc[start_date, field_names.TRANSFER]
        # Final balance after all transfers
        end_bal = df_sel.loc[end_date, field_names.BALANCE]

        # Get the transfers that went into / out of the specified accounts, and on which day (day 0 == start_date)
        mask = df_sel[field_names.TRANSFER] != 0.0
        trans_days = np.arange(len(df_sel))[mask]
        trans_amts = df_sel[field_names.TRANSFER].values[mask]
        num_days = (end_date - start_date).days

        # Determine the interest rate
        return calc_avg_interest_rate(start_bal=start_bal, end_bal=end_bal, num_days=num_days,
                                      trans_days=trans_days, trans_amts=trans_amts)

    def calc_avg_interest_rate(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                               as_ayr: bool = True, as_prcnt: bool = True) -> Optional[float]:
        """
        Compute the average interest rate over a specified date period.

        :param start_date: Optional[datetime]
            The first date to consider when computing the average interest rate. If not provided, use
            the first date in the MoneyFrame
        :param end_date: Optional[datetime]
            The last date to consider when computing the average interest rate. If not provided, use
            the final date in the MoneyFrame
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: Optional[float]
            The average interest rate over the time period. If it is not possible to compute, None is returned.
        """
        assert_type(start_date, datetime, optional=True)
        assert_type(end_date, datetime, optional=True)

        df = self.df.copy(deep=False)

        if start_date is not None and end_date is not None:
            start_date = coalesce(start_date, self.min_date())
            end_date = coalesce(end_date, self.max_date())
            assert start_date <= end_date, "start_date ({}) > end_date ({})".format(start_date, end_date)

            # Filter for the correct date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Get the start and end date available in the data
        start_date = df.iloc[0].name
        end_date = df.iloc[-1].name

        # Need at least two rows (days) of data to compute an interest rate
        if len(df) <= 1:
            return np.nan

        adr = self._calc_avg_interest_rate(df, start_date, end_date)
        return self._interest_rate_conversions(adr, as_ayr=as_ayr, as_prcnt=as_prcnt)

    def _calc_cum_interest_rate(self, as_ayr: bool = True, as_prcnt: bool = True) -> pd.Series:
        """
        Compute the cumulative interest rate over the period provided.

        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.Series
            Cumulative interest rate. The Series is indexed by date
        """

        start_date = self.min_date()
        adr = np.array([
            self._calc_avg_interest_rate(self.df, start_date=start_date, end_date=end_date)
            for end_date in self.df.index.get_level_values(field_names.DATE)
        ])
        return self._interest_rate_conversions(adr, as_ayr=as_ayr, as_prcnt=as_prcnt)

    @classmethod
    def from_updates_and_transfers(cls, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers,
                                   account_key: str):
        """
        Using both (1) periodic account balances, and (2) balance transfers to/from the account,
        create an instance of MoneyFrame for a specific account key. Dates between account updates
        are estimated by assuming a constant interest rate over the period.

        :param balance_updates: BalanceUpdates
            The DataSource containing all balance updates (not necessarily just for the account key specified)
        :param balance_transfers: BalanceTransfers
            The DataSource containing all balance transfers (not necessarily just for the account key specified)
        :param account_key: str
            Unique identifier for the account
        """
        assert_type(balance_updates, BalanceUpdates)
        assert_type(balance_transfers, BalanceTransfers)
        try:
            df = cls._est_daily_history_df(balance_updates, balance_transfers, account_key)
        except BalanceExtrapolationError as e:
            raise BalanceExtrapolationError(
                "Could not estimate daily account balances for account key '{}'.".format(account_key)
            ) from e

        return MoneyFrame(df)

    @classmethod
    def _est_daily_balances_between_updates(cls, start_date: datetime, end_date: datetime, start_bal: float,
                                            end_bal: float, transfers: pd.Series) -> pd.Series:
        """
        Estimate the daily balances between two balance updates, assuming a fixed rate of interest over the period.

        :param start_date: datetime
            The date of the first balance update
        :param end_date: datetime
            The date of the second balance update
        :param start_bal:
            The balance at the first update (at the end of the day, *after* any balance transfers)
        :param end_bal:
            The balance at the second update (at the end of the day, *after* any balance transfers)
        :param transfers:
            Series containing balance transfers to and from this account. Should be indexed by transfer date
        :return: pd.Series
            Estimated balance in the account each day during the period [start_date, end_date).
            The Series is indexed by date, and has values indicating the balance at the end of that
            day *after* any interest and balance transfers are applied.

        :raises TypeError: If any of the input types are invalid
        :raises ValueError: If any of the input values are invalid
        """

        # Check types (if incorrect, will raise a TypeError)
        assert_type(start_date, datetime)
        assert_type(end_date, datetime)

        # Check the input values
        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Invalid start/end dates have been passed: {} - {}".format(str(start_date), str(end_date)))

        log.debug("Updates from {} : {} -> {} : {}".format(start_date, start_bal, end_date, end_bal))

        # Get all the transfers from/to the account that happened between the period
        transfers = transfers[(transfers.index > start_date) & (transfers.index <= end_date)]
        transfer_days = (transfers.index - start_date).days
        transfer_amts = transfers.values
        num_days = (end_date - start_date).days

        # Estimate the interest rate during the period
        try:
            daily_rate = calc_avg_interest_rate(
                start_bal=start_bal, end_bal=end_bal, num_days=num_days, trans_days=transfer_days,
                trans_amts=transfer_amts
            )
        except NoSolutionFoundError as e:
            debug_str = "Trying to extrapolate balances between dates {} and {}.\n".format(start_date, end_date)
            debug_str += "The start and end balances are {} and {}.\n".format(start_bal, end_bal)
            debug_str += "Between these dates the following transfers were made:\n{}\n".format(transfers)
            raise BalanceExtrapolationError(debug_str) from e

        # Use the interest rate to estimate account balances on each day
        balances = calc_daily_balances_w_transfers(
            start_bal=start_bal, num_days=num_days, daily_rate=daily_rate,
            trans_amts=transfer_amts, trans_days=transfer_days
        )

        dates = dates_between(start_date, end_date)[:-1]
        return pd.Series(data=balances, index=dates)

    @classmethod
    def _est_daily_balances(cls, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers,
                            account_key: str) -> pd.Series:
        """
        Estimate the account balance on every day, assuming a fixed interest rate between each balance update.

        :param balance_updates: BalanceUpdates
            Dataset containing a history of account balance checkpoints e.g. what was the balance on a given day
        :param balance_transfers: BalanceTransfers
            Dataset containing a history of transfers between accounts
        :param account_key: str
             Unique string identifying an account

        :return: pd.Series
            Estimated balance in the account each day. The Series is indexed by date, and has a value
            indicating the balance at the end of that day *after* any interest and balance transfers are applied
        """
        log.debug("Getting balance updates and transfers")

        # Get balance updates and transfers for the specified account_key
        bal_updates = balance_updates.get_acc_updates(account_key=account_key)
        bal_transfers = balance_transfers.get_acc_transfers(account_key=account_key)

        # If there isn't any information, just return a DataFrame with today's date and zero balance
        if len(bal_updates) == 0 and len(bal_transfers) == 0:
            today = pd.to_datetime(datetime.now().date())
            return pd.Series(data=[0.0], index=[today], name=field_names.BALANCE).rename_axis(field_names.DATE)

        # If there are no balance updates, assume no interest, so that the latest balance update will be sum(transfers)
        if len(bal_updates) == 0:
            last_trans_date = bal_transfers.index[-1]
            bal_updates = pd.Series(data=[np.sum(bal_transfers)], index=[last_trans_date])

        if len(bal_transfers) > 0:
            # If the first balance transfer happens before the first update, assume that the transfer was the opening
            # payment, and the previous balance was zero
            first_update_date = bal_updates.index[0]
            first_bal_trans_date = bal_transfers.index[0]

            if first_bal_trans_date <= first_update_date:
                new_row = pd.Series(data=[0.0],index=[first_bal_trans_date - pd.Timedelta(days=1)])
                bal_updates = pd.concat([new_row, bal_updates]).sort_index()

            # If balance transfers happen after the last update, assume zero interest after the last update
            last_update_amt = bal_updates.values[-1]
            last_update_date = bal_updates.index[-1]
            last_trans_date = bal_transfers.index[-1]

            if last_trans_date > last_update_date:
                tot_transfers = bal_transfers.loc[bal_transfers.index > last_update_date].sum()
                new_row = pd.Series(data=[last_update_amt + tot_transfers],index=[last_trans_date])
                bal_updates = pd.concat([new_row, bal_updates]).sort_index()

        # If there is only one balance update, just return the single data point
        if len(bal_updates) == 1:
            return bal_updates

        # Extrapolate the daily balances between each update.
        # First reshape the data to obtain consecutive balance updates
        bal_updates_df = bal_updates.to_frame()
        bal_updates_df[field_names.END_BALANCE] = bal_updates.values
        bal_updates_df[field_names.END_DATE] = bal_updates.index.values
        bal_updates_df[field_names.START_BALANCE] = bal_updates_df[field_names.END_BALANCE].shift(1)
        bal_updates_df[field_names.START_DATE] = bal_updates_df[field_names.END_DATE].shift(1)
        bal_updates_df = bal_updates_df.dropna()

        log.debug("Looping over balance updates")
        series = [
            cls._est_daily_balances_between_updates(
                start_date=row[field_names.START_DATE],
                end_date=row[field_names.END_DATE],
                start_bal=row[field_names.START_BALANCE],
                end_bal=row[field_names.END_BALANCE],
                transfers=bal_transfers,
            )
            for _, row in bal_updates_df.iterrows()
        ]

        log.debug("Adding most recent balance update")
        series.append(bal_updates.iloc[-1:])

        log.debug("Returning daily balance updates")
        return pd.concat(series).sort_index().rename(field_names.BALANCE).rename_axis(field_names.DATE)

    @classmethod
    def _est_daily_history_df(cls, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers,
                              account_key: str) -> pd.DataFrame:
        """
        Get a daily summary of the account. It gives the account balance at the end of the day, as well
        as transfers into/from the account, and any interest payments.

        :param balance_updates: BalanceUpdates
            Dataset containing a history of account balance checkpoints e.g. what was the balance on a given day
        :param balance_transfers: BalanceTransfers
            Dataset containing a history of transfers between accounts
        :param account_key: str
             Unique string identifying an account
        :return: pd.DataFrame
            DataFrame containing the daily account summary. It is indexed by date, and has the following columns
                - interest
                - transfer
                - balance
        """
        log.debug("getting daily balances")
        daily_balances = cls._est_daily_balances(balance_updates, balance_transfers, account_key)

        log.debug("getting daily transfers")
        bal_transfers = balance_transfers.get_acc_transfers(account_key=account_key)

        daily_summary = pd.concat([daily_balances, bal_transfers], axis=1).fillna(0.0)

        log.debug("determining interest")
        daily_summary[field_names.INTEREST] = (
                daily_summary[field_names.BALANCE] -
                daily_summary[field_names.BALANCE].shift(1) -
                daily_summary[field_names.TRANSFER]
        ).fillna(0.0)

        return daily_summary

    @classmethod
    def from_sum(cls, daily_acc_hists: Iterable["MoneyFrame"]) -> "MoneyFrame":
        """
        Create MoneyFrame from a combination of other MoneyFrame instances.

        :param daily_acc_hists: Iterable["MoneyFrame"]
            An iterable collection of MoneyFrame objects

        :return: MoneyFrame
            A MoneyFrame that combines all instances given
        """

        daily_acc_hists = [x for x in daily_acc_hists]
        assert len(daily_acc_hists) != 0, \
            "MoneyFrame.from_sum() must be passed at least one MoneyFrame"

        # If we have any zombie accounts, they can be removed
        daily_acc_hists_active = [x for x in daily_acc_hists if not x.is_zombie()]

        # If we only have zombie accounts, no choice but to return one of them
        if len(daily_acc_hists_active) == 0:
            return daily_acc_hists[0]
        # If there is only one account, we can just return it
        if len(daily_acc_hists_active) == 1:
            return daily_acc_hists_active[0]

        # Find the mix/max date over all MoneyFrame instances
        ranges = [daily_acc_hist.minmax_date() for daily_acc_hist in daily_acc_hists_active]
        start_dates, end_dates = list(zip(*ranges))
        start_date = min(*start_dates)
        end_date = max(*end_dates)

        dfs = [
            daily_acc_hist.to_df(start_date=start_date, end_date=end_date)
            for daily_acc_hist in daily_acc_hists_active
        ]
        return MoneyFrame(reduce(lambda x, y: x + y, dfs))

    @classmethod
    def from_fixed_rate(cls, days: Union[int, Tuple[Any, Any]], start_bal: float,
                        ayr_prcnt: float) -> "MoneyFrame":
        """
        Create MoneyFrame for a specified date range, assuming a given starting balance, and
        a fixed interest rate.

        :param days: Union[int, Tuple[Any, Any]]
            Either the number of days, or a tuple containing the start/end date. For example:
                - ('2020-06-10', '2020-06-13')
                - 5 (will use a date range of 5 days, ending today)
        :param start_bal: float
            The starting balance e.g. £100.0
        :param ayr_prcnt: float
            The average yearly interest rate as a percentage e.g. 5%
        :return: MoneyFrame
        """
        if isinstance(days, int):
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=days - 1)
        elif isinstance(days, tuple) and len(days) == 2:
            start_date, end_date = days
        else:
            raise TypeError("Expected an integer of a tuple of dates")

        # Get all dates within the range
        dates = pd.date_range(start_date, end_date)
        num_days = len(dates)

        # Determine the balance each day, assuming fixed interest rate
        adr = ayr_to_adr(ayr_prcnt * 0.01)
        balances = calc_daily_balances_w_transfers(start_bal=start_bal, num_days=num_days, daily_rate=adr)

        # Return MoneyFrame
        df = pd.DataFrame({field_names.DATE: dates, field_names.BALANCE: balances})
        return MoneyFrame(df)

    @classmethod
    def create_empty(cls):
        return MoneyFrame(pd.DataFrame({field_names.DATE: [], field_names.BALANCE: []}))

    def __add__(self, other):
        return MoneyFrame.from_sum([self, other])

    def __eq__(self, other):
        return len(self) == len(other) and np.all(self.df.index == other.df.index) and \
               np.allclose(self.df.values, other.df.values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, x) -> "MoneyFrame":

        if isinstance(x, slice):
            assert x.step is None, "step parameter is not accepted when slicing a MoneyFrame"
            if isinstance(coalesce(x.start, 0), int) and isinstance(coalesce(x.stop, len(self)), int):
                start_ix, end_ix = coalesce(x.start, 0), coalesce(x.stop, len(self))
                return MoneyFrame(self.df.iloc[start_ix:end_ix])
            else:
                start_date = pd.to_datetime(coalesce(x.start, self.min_date()))
                end_date = pd.to_datetime(coalesce(x.stop, self.max_date()))
                return self.get_slice(start_date, end_date, extrapolate=False)
        elif isinstance(x, int):
            if x == -1:
                return MoneyFrame(self.df.iloc[x:])
            return MoneyFrame(self.df.iloc[x:x + 1])
        elif isinstance(x, str) and x in self.df.columns:
            return self.df[x].values
        else:
            date = pd.to_datetime(x)
            return self.get_slice(date, date, extrapolate=False)
