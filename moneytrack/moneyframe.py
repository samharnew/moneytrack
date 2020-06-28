from datetime import datetime, timedelta
from functools import reduce
from typing import Any, Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd
import logging

from .datasets import DataFields, BalanceUpdates, BalanceTransfers
from .utils import coalesce, assert_type, adr_to_ayr, calc_avg_interest_rate, calc_daily_balances_w_transfers, \
    dates_between, ayr_to_adr

log = logging.getLogger("moneyframe")


class MoneyFrame:
    """
    The daily history of a bank account. Contains information about:
        - Any transfers from/to the account each day
        - Any interest payments into the account each day
        - The balance of the account at the end of the day *after* interest payments and transfers
    """

    expected_cols = [
        DataFields.INTEREST,
        DataFields.TRANSFER,
        DataFields.BALANCE,
    ]

    def __init__(self, df: pd.DataFrame):
        """
        Create a MoneyFrame from a pandas DataFrame. The following columns can be included:

            - DataFields.BALANCE
            - DataFields.DATE
            - DataFields.INTEREST
            - DataFields.TRANSFER

        Additional columns may be provided, but will be ignored. If the DataFrame is already indexed by date,
        the DataFields.DATE field is not needed. The DataFields.TRANSFER field is assumed to be zero if not
        provided. If the DataFields.INTEREST field is not provided, it is determined by considering the balance
        from the previous day, and any balance transfers into / out of the account.

        :param df: pd.DataFrame
        """

        # Check the DataFrame has all the columns
        assert DataFields.BALANCE in df.columns, "DataFrame must have column {}".format(DataFields.BALANCE)

        # Make a copy to ensure the original data doesn't get changed.
        df = df.copy()

        # If transfers are not given, assume there are none
        if DataFields.TRANSFER not in df.columns:
            df[DataFields.TRANSFER] = 0.0

        # If daily interest is not given, calculate from daily balances and transfers
        if DataFields.INTEREST not in df.columns:
            df[DataFields.INTEREST] = (df[DataFields.BALANCE] - df[DataFields.TRANSFER] -
                                       df[DataFields.BALANCE].shift(1)).fillna(0.0)

        # If the date isn't already the index column, make sure it is
        if DataFields.DATE in df.columns:
            df = df.set_index(DataFields.DATE)

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
            dates = pd.DataFrame({DataFields.DATE: pd.date_range(start_date, end_date)}).set_index(DataFields.DATE)
            df = dates.join(self.df, how="outer").sort_index(ascending=True)

            # Extrapolate first/last balances backwards/forwards
            df[DataFields.BALANCE] = df[DataFields.BALANCE].ffill().bfill()

            # Inner join to select only the range needed. Fill in missing transfer/interest values with 0.
            df = df.loc[start_date:end_date].fillna(0.0)
        else:
            df = self.df.loc[start_date:end_date]

        return MoneyFrame(df)

    def __getitem__(self, x) -> "MoneyFrame":

        if isinstance(x, slice):
            assert x.step is None, "step parameter is not accepted when slicing MoneyFrame"
            if isinstance(coalesce(x.start, 0), int) and isinstance(coalesce(x.stop, -1), int):
                start_ix, end_ix = coalesce(x.start, 0), coalesce(x.stop, -1)
                dah = MoneyFrame(self.df.iloc[start_ix:end_ix])
            else:
                start_date = pd.to_datetime(coalesce(x.start, self.min_date()))
                end_date = pd.to_datetime(coalesce(x.stop, self.max_date()))
                assert start_date <= end_date, "Must have start_date <= end_date"
                dah = self.get_slice(start_date, end_date, extrapolate=False)
        elif isinstance(x, int):
            dah = MoneyFrame(self.df.iloc[x:x + 1])
        else:
            date = pd.to_datetime(x)
            dah = self.get_slice(date, date, extrapolate=False)

        return dah

    def __len__(self):
        return len(self.df)

    def to_df(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
              inc_interest_rate = False, inc_cum_interest_rate: bool = False, as_ayr: bool = True,
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
                - DataFields.INTEREST
                - DataFields.TRANSFER
                - DataFields.BALANCE
                - DataFields.ACCOUNT_KEY [optional]
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
            df[DataFields.INTEREST_RATE] = self._calc_interest_rate(as_ayr=as_ayr, as_prcnt=as_prcnt)
        if inc_cum_interest_rate:
            df[DataFields.CUM_INTEREST_RATE] = self._calc_cum_interest_rate(as_ayr=as_ayr, as_prcnt=as_prcnt)
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
        return self.to_df(**kwargs)[DataFields.INTEREST]

    def get_daily_balance(self, **kwargs) -> pd.Series:
        """Account balance on each day *after* any interest and balance transfers are applied, indexed by date"""
        return self.to_df(**kwargs)[DataFields.BALANCE]

    def get_daily_transfers(self, **kwargs) -> pd.Series:
        """Transfers in/out of the account each day, indexed by date."""
        return self.to_df(**kwargs)[DataFields.TRANSFER]

    def get_daily_interest_rate(self, **kwargs) -> pd.Series:
        """Series of the daily interest rate, indexed by date."""
        return self.to_df(inc_interest_rate=True, **kwargs)[DataFields.INTEREST_RATE]

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
        ir = ir*100.0 if as_prcnt else ir
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
        adr = self.df[DataFields.INTEREST]/self.df[DataFields.BALANCE].shift(1)
        return self._interest_rate_conversions(adr, as_ayr, as_prcnt)

    @classmethod
    def _calc_avg_interest_rate(cls, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
        """
        Calculate the average interest rate between two dates.

        :param df: pd.DataFrame
            Pandas DataFrame indexed by date with columns DataFields.BALANCE, DataFields.INTEREST, DataFields.TRANSFERS
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
        start_bal = df_sel.loc[start_date, DataFields.BALANCE] - df_sel.loc[start_date, DataFields.TRANSFER]
        # Final balance after all transfers
        end_bal = df_sel.loc[end_date, DataFields.BALANCE]

        # Get the transfers that went into / out of the specified accounts, and on which day (day 0 == start_date)
        mask = df_sel[DataFields.TRANSFER] != 0.0
        trans_days = np.arange(len(df_sel))[mask]
        trans_amts = df_sel[DataFields.TRANSFER].values[mask]
        num_days = (end_date - start_date).days

        # Determine the interest rate
        return calc_avg_interest_rate(start_bal=start_bal, end_bal=end_bal, num_days=num_days,
                                      trans_days=trans_days, trans_amts=trans_amts)

    def calc_avg_interest_rate(self, start_date: Optional[datetime], end_date: Optional[datetime],
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
            for end_date in self.df.index.get_level_values(DataFields.DATE)
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

        df = cls._est_daily_history_df(balance_updates, balance_transfers, account_key)
        return MoneyFrame(df)

    @classmethod
    def _est_daily_balances_between_updates(cls, start_date: datetime, end_date: datetime, start_bal: float,
                                            end_bal: float, bal_transfers_df: pd.DataFrame) -> pd.DataFrame:
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
        :param bal_transfers_df:
            DataFrame containing balance transfers to and from this account. The DataFrame should have the columns
            DataField.DATE and DataField.AMOUNT
        :return: pd.DataFrame
            Estimated balance in the account each day during the period [start_date, end_date).
            The DataFrame has columns DataField.DATE and DataField.BALANCE indicating the balance at the end of that
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
        transfers = bal_transfers_df[(bal_transfers_df[DataFields.DATE] > start_date) &
                                     (bal_transfers_df[DataFields.DATE] <= end_date)]
        trans_days = (transfers[DataFields.DATE] - start_date).dt.days
        num_days = (end_date - start_date).days
        trans_amts = transfers[DataFields.AMOUNT]

        # Estimate the interest rate during the period
        log.debug("Calculating daily interest rate")
        daily_rate = calc_avg_interest_rate(
            start_bal=start_bal, end_bal=end_bal, num_days=num_days, trans_days=trans_days, trans_amts=trans_amts
        )

        # Use the interest rate to estimate account balances on each day
        log.debug("Daily interest rate is {}".format(daily_rate))
        balances = calc_daily_balances_w_transfers(
            start_bal=start_bal, num_days=num_days, daily_rate=daily_rate, trans_amts=trans_amts, trans_days=trans_days
        )

        dates = dates_between(start_date, end_date)[:-1]
        return pd.DataFrame({DataFields.DATE: dates, DataFields.BALANCE: balances})

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
        bal_updates_df = balance_updates.get_acc_updates(account_key=account_key)\
            .sort_values(DataFields.DATE, ascending=True)
        bal_transfers_df = balance_transfers.get_acc_transfers(account_key=account_key)\
            .sort_values(DataFields.DATE, ascending=True)

        # If there isn't any information, just return a DataFrame with today's date and zero balance
        if len(bal_updates_df) == 0 and len(bal_transfers_df) == 0:
            today = pd.to_datetime(datetime.now().date())
            return pd.DataFrame(
                [{DataFields.BALANCE: 0.0, DataFields.DATE: today}]
            ).set_index(DataFields.DATE)[DataFields.BALANCE]

        # If there are no balance updates, assume no interest, so that the latest balance update will be sum(transfers)
        if len(bal_updates_df) == 0:
            last_bal_trans_date = bal_transfers_df[DataFields.DATE].iloc[-1]
            bal_updates_df = pd.DataFrame([{
                DataFields.DATE: last_bal_trans_date,
                DataFields.BALANCE: bal_transfers_df[DataFields.AMOUNT].sum()
            }])

        if len(bal_transfers_df) > 0:
            # If the first balance transfer happens before the first update, assume that the transfer was the opening
            # payment, and the previous balance was zero
            first_update_date = bal_updates_df[DataFields.DATE].iloc[0]
            first_bal_trans_date = bal_transfers_df[DataFields.DATE].iloc[0]

            if first_bal_trans_date <= first_update_date:
                new_row = {
                    DataFields.DATE: first_bal_trans_date - pd.Timedelta(days=1),
                    DataFields.BALANCE: 0.0
                }
                bal_updates_df = bal_updates_df.append(pd.DataFrame([new_row]), ignore_index=True)
                bal_updates_df.sort_values(DataFields.DATE, ascending=True, inplace=True)

            # If balance transfers happen after the last update, assume zero interest after the last update
            last_update_amt = bal_updates_df[DataFields.BALANCE].iloc[-1]
            last_update_date = bal_updates_df[DataFields.DATE].iloc[-1]
            last_bal_trans_date = bal_transfers_df[DataFields.DATE].iloc[-1]

            if last_bal_trans_date > last_update_date:
                tot_transfers = bal_transfers_df.loc[
                    bal_transfers_df[DataFields.DATE] > last_update_date, DataFields.AMOUNT
                ].sum()
                new_row = {
                    DataFields.DATE: last_bal_trans_date,
                    DataFields.BALANCE: last_update_amt+tot_transfers
                }
                bal_updates_df = bal_updates_df.append(pd.DataFrame([new_row]), ignore_index=True)
                bal_updates_df.sort_values(DataFields.DATE, ascending=True, inplace=True)

        # If there is only one balance update, just return the single data point
        if len(bal_updates_df) == 1:
            return bal_updates_df.set_index(DataFields.DATE)[DataFields.BALANCE]

        # Extrapolate the daily balances between each update.
        # First reshape the data to obtain consecutive balance updates
        bal_updates_df[DataFields.START_BALANCE] = bal_updates_df[DataFields.BALANCE].shift(1)
        bal_updates_df[DataFields.START_DATE] = bal_updates_df[DataFields.DATE].shift(1)
        bal_updates_df[DataFields.END_BALANCE] = bal_updates_df[DataFields.BALANCE]
        bal_updates_df[DataFields.END_DATE] = bal_updates_df[DataFields.DATE]
        bal_updates_df = bal_updates_df.dropna()

        log.debug("Looping over balance updates")
        dfs = [
            cls._est_daily_balances_between_updates(
                start_date=row[DataFields.START_DATE],
                end_date=row[DataFields.END_DATE],
                start_bal=row[DataFields.START_BALANCE],
                end_bal=row[DataFields.END_BALANCE],
                bal_transfers_df=bal_transfers_df,
            )
            for _, row in bal_updates_df.iterrows()
        ]

        log.debug("Adding most recent balance update")
        dfs.append(bal_updates_df.tail(1)[[DataFields.DATE, DataFields.BALANCE]])

        log.debug("Returning daily balance updates")
        return pd.concat(dfs).sort_values(DataFields.DATE).set_index(DataFields.DATE)[DataFields.BALANCE]

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
        bal_transfers_df = balance_transfers.get_acc_transfers(account_key=account_key)
        bal_transfers_df = bal_transfers_df.groupby(DataFields.DATE)[DataFields.AMOUNT].sum()
        daily_trans = bal_transfers_df.rename(DataFields.TRANSFER)

        daily_summary = pd.concat([daily_balances, daily_trans], axis=1).fillna(0.0)

        log.debug("determining interest")
        daily_summary[DataFields.INTEREST] = (
                daily_summary[DataFields.BALANCE] -
                daily_summary[DataFields.BALANCE].shift(1) -
                daily_summary[DataFields.TRANSFER]
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
            start_date = end_date - timedelta(days=days-1)
        elif isinstance(days, tuple) and len(days) == 2:
            start_date, end_date = days
        else:
            raise TypeError("Expected an integer of a tuple of dates")

        # Get all dates within the range
        dates = pd.date_range(start_date, end_date)
        num_days = len(dates)

        # Determine the balance each day, assuming fixed interest rate
        adr = ayr_to_adr(ayr_prcnt*0.01)
        balances = calc_daily_balances_w_transfers(start_bal=start_bal, num_days=num_days, daily_rate=adr)

        # Return MoneyFrame
        df = pd.DataFrame({DataFields.DATE: dates, DataFields.BALANCE: balances})
        return MoneyFrame(df)