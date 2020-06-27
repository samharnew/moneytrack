from moneytrack import DataFields, Accounts, BalanceUpdates, BalanceTransfers, calc_avg_interest_rate, \
    calc_daily_balances_w_transfers, dates_between, adr_to_ayr, assert_type, Config

import pandas as pd
import numpy as np
import logging
import glob
import os
import datetime
from functools import reduce

from typing import Optional, Union, List, Iterable, Tuple, Dict

log = logging.getLogger("core")


class DailyAccountHistory:
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

    def __init__(self, daily_acc_hist_df: pd.DataFrame):
        """
        :param daily_acc_hist_df: pd.DataFrame
            DataFrame containing the daily account history. It must be indexed by date, and have the following columns:
                - DataFields.INTEREST
                - DataFields.TRANSFER
                - DataFields.BALANCE
        """

        # Check the DataFrame has all the columns
        for col in self.expected_cols:
            assert col in daily_acc_hist_df.columns, "DataFrame must have column {}".format(col)

        # Make a copy to ensure the original data doesn't get changed.
        df = daily_acc_hist_df.copy()

        # If the data isn't already the index column, make sure it is
        if DataFields.DATE in df.columns:
            df.set_index(DataFields.DATE)

        # Make sure we only have the columns needed
        df = df[self.expected_cols]

        # Check the data types are legit
        try:
            df = df.astype(float, copy=True)
        except Exception:
            raise ValueError("Some columns of the DataFrame are not compatible with numeric types")

        # Check for missing values
        assert df.isnull().values.any() == False, "DataFrame contains missing values"

        # Check that the date index is complete
        missing_dates = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
        assert len(missing_dates) == 0, "Daily history should contain all dates between the start and end date"

        self.df = df.sort_index(ascending=True)

    def get_df(self, start_date: Optional[pd.datetime] = None, end_date: Optional[pd.datetime] = None,
               inc_interest_rate = False, inc_cum_interest_rate: bool = False, as_ayr: bool = True,
               as_prcnt: bool = True) -> pd.DataFrame:
        """
        Get pandas DataFrame containing the daily account history. Gives the account balance at the end of the day,
        as well as transfers into/from the account, and any interest payments.

        :param start_date: pd.datetime
            Start the daily summary on this date
        :param end_date: pd.datetime
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
        assert_type(start_date, pd.datetime, optional=True)
        assert_type(end_date, pd.datetime, optional=True)

        df = self.df.copy(deep=False)

        if start_date is not None or end_date is not None:

            # If a date isn't provided for either start/end, use the first/last date available.
            end_date = end_date if end_date is not None else df.index.max()
            start_date = start_date if start_date is not None else df.index.min()

            # Outer join a list of all dates in the range to fill in any gaps
            dates = pd.DataFrame({DataFields.DATE: dates_between(start_date, end_date)}).set_index(DataFields.DATE)
            df = dates.join(df, how="outer").sort_index(ascending=True)

            # Extrapolate first/last balances backwards/forwards
            df[DataFields.BALANCE] = df[DataFields.BALANCE].ffill().bfill()

            # Inner join to select only the range needed. Fill in missing transfer/interest values with 0.
            df = dates.join(df, how="inner").fillna(0.0)

        if inc_interest_rate:
            df[DataFields.INTEREST_RATE] = self._calc_interest_rate(df, as_ayr=as_ayr, as_prcnt=as_prcnt)
        if inc_cum_interest_rate:
            df[DataFields.CUM_INTEREST_RATE] = self._calc_cum_interest_rate(df, as_ayr=as_ayr, as_prcnt=as_prcnt)

        return df

    def is_zombie(self, **kwargs):
        """The account is a zombie if it has always had a balance of zero"""
        return (self.get_daily_balance(**kwargs) == 0.0).values.all()

    def date_range(self) -> Tuple[pd.datetime, pd.datetime]:
        """Return a tuple containing the min/max date"""
        df = self.get_df()
        return df.index.min(), df.index.max()

    def get_daily_interest(self, **kwargs) -> pd.Series:
        """Amount of interest added to the account each day, indexed by date"""
        return self.get_df(**kwargs)[DataFields.INTEREST]

    def get_daily_balance(self, **kwargs) -> pd.Series:
        """Account balance on each day *after* any interest and balance transfers are applied, indexed by date"""
        return self.get_df(**kwargs)[DataFields.BALANCE]

    def get_daily_transfers(self, **kwargs) -> pd.Series:
        """Transfers in/out of the account each day, indexed by date."""
        return self.get_df(**kwargs)[DataFields.TRANSFER]

    def get_daily_interest_rate(self, **kwargs) -> pd.Series:
        """Series of the daily interest rate, indexed by date."""
        return self.get_df(inc_interest_rate=True, **kwargs)[DataFields.INTEREST_RATE]

    @classmethod
    def _interest_rate_conversions(cls, adr: np.array, as_ayr: bool = True, as_prcnt: bool = True) -> np.array:
        """
        Convert a fractional average daily interest rate into (optionally) an average yearly rate (opionally) as
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

    @classmethod
    def _calc_interest_rate(cls, df: pd.DataFrame, as_ayr: bool = True, as_prcnt: bool = True) -> pd.Series:
        """
        Calculate the daily interest rate.

        :param df: pd.DataFrame
            DataFrame of the daily account history
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.Series
            Daily interest rate. The Series is indexed by date
        """
        adr = df[DataFields.INTEREST]/df[DataFields.BALANCE].shift(1)
        return cls._interest_rate_conversions(adr, as_ayr, as_prcnt)

    @classmethod
    def _calc_avg_interest_rate(cls, df: pd.DataFrame, start_date: pd.datetime, end_date: pd.datetime) -> float:
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

        if num_days == 0:
            return np.nan

        # Determine the interest rate
        return calc_avg_interest_rate(start_bal=start_bal, end_bal=end_bal, num_days=num_days,
                                      trans_days=trans_days, trans_amts=trans_amts)

    def calc_avg_interest_rate(self, start_date: Optional[pd.datetime], end_date: Optional[pd.datetime],
                               as_ayr: bool = True, as_prcnt: bool = True) -> Optional[float]:
        """
        Compute the average interest rate over a specified date period.

        :param start_date: Optional[pd.datetime]
            The first date to consider when computing the average interest rate. If not provided, use
            the first date in the DailyAccountHistory
        :param end_date: Optional[pd.datetime]
            The last date to consider when computing the average interest rate. If not provided, use
            the final date in the DailyAccountHistory
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: Optional[float]
            The average interest rate over the time period. If it is not possible to compute, None is returned.
        """
        assert_type(start_date, pd.datetime, optional=True)
        assert_type(end_date, pd.datetime, optional=True)

        df = self.df.copy(deep=False)

        if start_date is not None and end_date is not None:
            start_date = df.iloc[0].name if start_date is None else start_date
            end_date = df.iloc[-1].name if end_date is None else end_date
            # Filter for the correct date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        if len(df) <= 1:
            return None

        # Get the start and end date available in the data
        start_dt = df.iloc[0].name
        end_dt = df.iloc[-1].name

        adr = self._calc_avg_interest_rate(df, start_dt, end_dt)

        return self._interest_rate_conversions(adr, as_ayr=as_ayr, as_prcnt=as_prcnt)

    @classmethod
    def _calc_cum_interest_rate(cls, df: pd.DataFrame, as_ayr: bool = True, as_prcnt: bool = True) -> pd.Series:
        """
        Compute the cumulative interest rate over the period provided.

        :param df: pd.DataFrame
            Pandas DataFrame indexed by date with columns DataFields.BALANCE, DataFields.INTEREST, DataFields.TRANSFERS
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.Series
            Cumulative interest rate. The Series is indexed by date
        """

        start_date = df.index.min()
        adr = np.array([
            cls._calc_avg_interest_rate(df, start_date=start_date, end_date=end_date)
            for end_date in df.index.get_level_values(DataFields.DATE)
        ])
        return cls._interest_rate_conversions(adr, as_ayr=as_ayr, as_prcnt=as_prcnt)

    @classmethod
    def from_updates_and_transfers(cls, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers,
                                   account_key: str):
        """
        Using both (1) periodic account balances, and (2) balance transfers to/from the account,
        create an instance of DailyAccountHistory for a specific account key. Dates between account updates
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
        return DailyAccountHistory(df)

    @classmethod
    def _est_daily_balances_between_updates(cls, start_date: pd.datetime, end_date: pd.datetime, start_bal: float,
                                            end_bal: float, bal_transfers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the daily balances between two balance updates, assuming a fixed rate of interest over the period.

        :param start_date: pd.datetime
            The date of the first balance update
        :param end_date: pd.datetime
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
        assert_type(start_date, pd.datetime)
        assert_type(end_date, pd.datetime)

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
            today = pd.to_datetime(datetime.datetime.now().date())
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
    def from_sum(cls, daily_acc_hists: Iterable["DailyAccountHistory"]) -> "DailyAccountHistory":
        """
        Create DailyAccountHistory from a combination of other DailyAccountHistory instances.

        :param daily_acc_hists: Iterable["DailyAccountHistory"]
            An iterable collection of DailyAccountHistory objects

        :return: DailyAccountHistory
            A DailyAccountHistory that combines all instances given
        """

        daily_acc_hists = [x for x in daily_acc_hists]
        assert len(daily_acc_hists) != 0, \
            "DailyAccountHistory.from_sum() must be passed at least one DailyAccountHistory"

        # If we have any zombie accounts, they can be removed
        daily_acc_hists_active = [x for x in daily_acc_hists if not x.is_zombie()]

        # If we only have zombie accounts, no choice but to return one of them
        if len(daily_acc_hists_active) == 0:
            return daily_acc_hists[0]
        # If there is only one account, we can just return it
        if len(daily_acc_hists_active) == 1:
            return daily_acc_hists_active[0]

        # Find the mix/max date over all DailyAccountHistory instances
        ranges = [daily_acc_hist.date_range() for daily_acc_hist in daily_acc_hists_active]
        start_dates, end_dates = list(zip(*ranges))
        start_date = min(*start_dates)
        end_date = max(*end_dates)

        dfs = [
            daily_acc_hist.get_df(start_date=start_date, end_date=end_date)
            for daily_acc_hist in daily_acc_hists_active
        ]
        return DailyAccountHistory(reduce(lambda x,y: x+y, dfs))


class MoneyData:
    """
    Container for all information about account portfolio:
        - Accounts: Details of each account in portfolio
        - Balance Updates: A checkpoint of an account balance at a point in time
        - Balance Transfers: Transfers between accounts, or from external sources
    """

    def __init__(self, accounts: Accounts, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers):
        """
        Create a MoneyData instance.

        :param accounts: Accounts
            Dataset containing details of each account in portfolio
        :param balance_updates: BalanceUpdates
            Dataset containing a history of account balance checkpoints e.g. what was the balance on a given day
        :param balance_transfers: BalanceTransfers
            Dataset containing a history of transfers between accounts
        """
        self.accounts = accounts
        self.balance_updates = balance_updates
        self.balance_transfers = balance_transfers

        # Create a daily account history for each account
        self.daily_acc_hist_dict = {
            acc_key: DailyAccountHistory.from_updates_and_transfers(
                self.balance_updates,
                self.balance_transfers,
                acc_key
            )
            for acc_key in accounts.get_account_keys()
        }

    def filter_acc_hist(self, filters: Optional[Dict[str, Union[str, List[str]]]]) -> \
            Dict[str, DailyAccountHistory]:
        """
        Filter the dictionary containing the account histories.

        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {DataFields.ACCOUNT_KEY: "TSB"}
                - {DataFields.ACCOUNT_KEY: ["TSB", "LLOYDS"], DataFields.ACCOUNT_TYPE: "CURRENT"}
        :return: Dict[str, DailyAccountHistory]
            A filtered dictionary
        """
        if filters is None:
            return self.daily_acc_hist_dict
        acc_keys = self.accounts.get_matching_account_keys(filters, without_ext=True)
        return {
            acc_key: acc_hist
            for acc_key, acc_hist in self.daily_acc_hist_dict.items() if acc_key in acc_keys
        }

    def aggregate_acc_hist(self, filters: Optional[Dict[str, Union[str, List[str]]]], agg: Union[bool, str] = True)\
            -> Tuple[Dict[str, DailyAccountHistory], str]:
        """
        Aggregate the dictionary containing account histories. Can be aggregated by any column in the Accounts data
        e.g. passing agg=DataFields.ACCOUNT_TYPE will return a dictionary with keys indicating the account type, and
        values containing the DailyAccountHistory for all accounts of that type.

        Alternatively False can be passed to do no aggregation (i.e. aggregated by DataFields.ACCOUNT_KEY), or True
        can be passed to aggregate ALL accounts.

        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {DataFields.ACCOUNT_KEY: "TSB"}
                - {DataFields.ACCOUNT_KEY: ["TSB", "LLOYDS"], DataFields.ACCOUNT_TYPE: "CURRENT"}
        :param agg: Union[bool, str]
            How to aggregate the data e.g.
                - DataFields.ACCOUNT_TYPE: will aggregate by account type
                - False: will perform no aggregation
                - True: will aggregate all the accounts
        :return: Tuple[Dict[str, DailyAccountHistory], str]
            A tuple containing:
                - Dictionary containing the aggregated DailyAccountHistory
                - The aggregation level
        """

        hist_dict = self.filter_acc_hist(filters)

        if agg is True:
            agg_dict = {"ALL": DailyAccountHistory.from_sum(hist_dict.values())}
            return agg_dict, "ALL"
        if agg is False or agg.upper() == DataFields.ACCOUNT_KEY.upper():
            return hist_dict, DataFields.ACCOUNT_KEY

        df_acc = self.accounts.get_df().copy()
        df_acc.columns = [c.upper() for c in df_acc.columns]
        df_acc = df_acc[[DataFields.ACCOUNT_KEY.upper(), agg.upper()]]
        df_acc = df_acc[df_acc[DataFields.ACCOUNT_KEY].isin(hist_dict.keys())]
        return {
            agg_name: DailyAccountHistory.from_sum(
                [acc_hist for acc_key, acc_hist in hist_dict.items()
                 if acc_key in df_g[DataFields.ACCOUNT_KEY.upper()].values]
            )
            for agg_name, df_g in df_acc.groupby(agg.upper())
        }, agg

    def get_daily_account_history_df(self, filters: Optional[Dict[str, Union[str, List[str]]]] = None,
                                     agg: Union[bool, str] = True, ret_agg_lvl: bool = False,
                                     start_date: Optional[pd.datetime] = None, end_date: Optional[pd.datetime] = None,
                                     inc_cum_interest_rate: bool = False, inc_interest_rate: bool = False,
                                     as_ayr: bool = True, as_prcnt: bool = True) -> \
            Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """
        Get a pandas DataFrame containing the daily account history for one or multiple accounts.

        :param start_date: pd.datetime
            Start the daily summary on this date
        :param end_date: pd.datetime
            End the daily summary on this date
        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {DataFields.ACCOUNT_KEY: "TSB"}
                - {DataFields.ACCOUNT_KEY: ["TSB", "LLOYDS"], DataFields.ACCOUNT_TYPE: "CURRENT"}
        :param agg: Union[bool, str]
            How to aggregate the data e.g.
                - DataFields.ACCOUNT_TYPE: will aggregate by account type
                - False: will perform no aggregation
                - True: will aggregate all the accounts
        :param ret_agg_lvl: bool
            Optionally return the aggregation level along with the
        :param inc_interest_rate: bool
            Include a column containing the interest rate
        :param inc_cum_interest_rate: bool
            Include a column containing the cumulative interest rate
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.DataFrame
            A DataFrame containing the daily account history from the specified accounts. If agg = False, the
            DataFrame will be indexed by date and the aggregation level. If agg = True, the DataFrame will be indexed
            only by 'date'
        """

        hist_dict, agg_lvl = self.aggregate_acc_hist(filters, agg)

        dfs = [
            daily_acc_hist.get_df(
                start_date=start_date,
                end_date=end_date,
                inc_cum_interest_rate=inc_cum_interest_rate,
                inc_interest_rate=inc_interest_rate,
                as_ayr=as_ayr,
                as_prcnt=as_prcnt,
            ).assign(**{agg_lvl: agg_val})
            for agg_val, daily_acc_hist in hist_dict.items()
        ]

        df = pd.concat(dfs).reset_index().set_index([DataFields.DATE, agg_lvl])
        return (df, agg_lvl) if ret_agg_lvl else df

    def interest_rate_breakdown(self, start_date: pd.datetime, end_date: pd.datetime, agg: bool = False,
                                filters = None, as_ayr: bool = True, as_prcnt: bool = True) -> pd.DataFrame:
        """
        Get the average interest rate for each account, or alternatively over each category specified by the
        aggregation level.

        :param start_date: pd.datetime
            Start the daily summary on this date
        :param end_date: pd.datetime
            End the daily summary on this date
        :param agg: bool
            Aggregate the results together
        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {DataFields.ACCOUNT_KEY: "TSB"}
                - {DataFields.ACCOUNT_KEY: ["TSB", "LLOYDS"], DataFields.ACCOUNT_TYPE: "CURRENT"}
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: pd.DataFrame
            DataFrame containing the interest rate breakdown over the period specified
        """
        hist_dict, agg_lvl = self.aggregate_acc_hist(filters, agg)

        # Get daily account history
        interest_breakdown = [
            {
                agg_lvl: name,
                DataFields.INTEREST_RATE: daily_acc_hist.calc_avg_interest_rate(
                    start_date=start_date,
                    end_date=end_date,
                    as_prcnt=as_prcnt,
                    as_ayr=as_ayr,
                )
            }
            for name, daily_acc_hist in hist_dict.items()
        ]

        return pd.DataFrame(interest_breakdown).dropna().set_index(agg_lvl).sort_values(DataFields.INTEREST_RATE)

    @classmethod
    def from_csv(cls, accounts_path: str, balance_updates_path: str, balance_transfers_path: str) -> "MoneyData":
        """
        Helper method to create a MoneyData instance from csv files.
        :param accounts_path: str
            Path to csv file containing Accounts
        :param balance_updates_path: str
            Path to csv file containing BalanceUpdates
        :param balance_transfers_path: str
            Path to csv file containing BalanceTransfers
        :return: MoneyData
        """
        return MoneyData(
            Accounts.from_csv(accounts_path),
            BalanceUpdates.from_csv(balance_updates_path),
            BalanceTransfers.from_csv(balance_transfers_path),
        )

    @classmethod
    def find_path(cls, dir: str, filename_substr: str, file_ext: str = Config.CSV_EXT) -> str:
        """
        Find the path to a file given:
        :param dir: str
            Directory containing file
        :param filename_substr:
            A sub-string that will be contained within the filename
        :param file_ext:
            The extension of the file
        :return: str
            Path to the matching file
        :raises: AssertionError
            If either no file paths, or multiple file paths are found matching the criteria
        """
        files = glob.glob(os.path.join(dir, '*.' + file_ext.strip(".")))
        files = map(lambda x: os.path.split(x), files)
        matching_files = list(filter(lambda x: filename_substr in x[1], files))
        assert len(matching_files) != 0, "No {} files containing '{}' can be found".format(file_ext, filename_substr)
        assert len(matching_files) == 1, "More than one {} file containing '{}' has been found: {}".format(
            dir, filename_substr, "\n".join(matching_files)
        )
        return os.path.join(*matching_files[0])

    @classmethod
    def from_excel(cls, path: str, accounts_sheet="accounts", transfers_sheet="transfers",
                   updates_sheet="balance_updates") -> "MoneyData":
        """
        Helper method to create a MoneyData instance from a excel files containing multiple sheets

        :param path: str
            Path the excel file
        :param accounts_sheet: str
            Name of the excel sheet that contains the accounts
        :param transfers_sheet: str
            Name of the excel sheet that contains the balance transfers
        :param updates_sheet: str
            Name of the excel sheet that contains the balance updates
        :return: MoneyData
        """
        return MoneyData(
            Accounts.from_excel(path, accounts_sheet),
            BalanceUpdates.from_excel(path, updates_sheet),
            BalanceTransfers.from_excel(path, transfers_sheet),
        )

    @classmethod
    def from_csv_dir(cls, dir: str, accounts_substr="accounts", transfers_substr="transfers",
                     updates_substr="updates") -> "MoneyData":
        """
        Helper method to create a MoneyData instance from a directory containing csv files.

        :param dir: str
            Directory containing csv files
        :param accounts_substr: str
            Substring that will appear in the accounts filename
        :param transfers_substr: str
            Substring that will appear in the balance transfers filename
        :param updates_substr: str
            Substring that will appear in the balance updates filename
        :return: MoneyData
        """
        return cls.from_csv(
            accounts_path=cls.find_path(dir=dir, filename_substr=accounts_substr, file_ext=Config.CSV_EXT),
            balance_transfers_path=cls.find_path(dir=dir, filename_substr=transfers_substr, file_ext=Config.CSV_EXT),
            balance_updates_path=cls.find_path(dir=dir, filename_substr=updates_substr, file_ext=Config.CSV_EXT),
        )
