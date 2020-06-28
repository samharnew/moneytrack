import pandas as pd
import logging
import glob
import os

from datetime import datetime
from typing import Optional, Union, List, Tuple, Dict

from .datasets import DataFields, Accounts, BalanceUpdates, BalanceTransfers, Config
from .moneyframe import MoneyFrame

log = logging.getLogger("core")


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

        # Create a daily account history for each account
        self.daily_acc_hist_dict = {
            acc_key: MoneyFrame.from_updates_and_transfers(
                balance_updates,
                balance_transfers,
                acc_key
            )
            for acc_key in accounts.get_account_keys()
        }

    def filter_acc_hist(self, filters: Optional[Dict[str, Union[str, List[str]]]]) -> \
            Dict[str, MoneyFrame]:
        """
        Filter the dictionary containing the account histories.

        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {DataFields.ACCOUNT_KEY: "TSB"}
                - {DataFields.ACCOUNT_KEY: ["TSB", "LLOYDS"], DataFields.ACCOUNT_TYPE: "CURRENT"}
        :return: Dict[str, MoneyFrame]
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
            -> Tuple[Dict[str, MoneyFrame], str]:
        """
        Aggregate the dictionary containing account histories. Can be aggregated by any column in the Accounts data
        e.g. passing agg=DataFields.ACCOUNT_TYPE will return a dictionary with keys indicating the account type, and
        values containing the MoneyFrame for all accounts of that type.

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
        :return: Tuple[Dict[str, MoneyFrame], str]
            A tuple containing:
                - Dictionary containing the aggregated MoneyFrame
                - The aggregation level
        """

        hist_dict = self.filter_acc_hist(filters)

        if agg is True:
            agg_dict = {"ALL": MoneyFrame.from_sum(hist_dict.values())}
            return agg_dict, "ALL"
        if agg is False or agg.upper() == DataFields.ACCOUNT_KEY.upper():
            return hist_dict, DataFields.ACCOUNT_KEY

        df_acc = self.accounts.get_df().copy()
        df_acc.columns = [c.upper() for c in df_acc.columns]
        df_acc = df_acc[[DataFields.ACCOUNT_KEY.upper(), agg.upper()]]
        df_acc = df_acc[df_acc[DataFields.ACCOUNT_KEY].isin(hist_dict.keys())]
        return {
            agg_name: MoneyFrame.from_sum(
                [acc_hist for acc_key, acc_hist in hist_dict.items()
                 if acc_key in df_g[DataFields.ACCOUNT_KEY.upper()].values]
            )
            for agg_name, df_g in df_acc.groupby(agg.upper())
        }, agg

    def get_daily_account_history_df(self, filters: Optional[Dict[str, Union[str, List[str]]]] = None,
                                     agg: Union[bool, str] = True, ret_agg_lvl: bool = False,
                                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                     inc_cum_interest_rate: bool = False, inc_interest_rate: bool = False,
                                     as_ayr: bool = True, as_prcnt: bool = True) -> \
            Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """
        Get a pandas DataFrame containing the daily account history for one or multiple accounts.

        :param start_date: datetime
            Start the daily summary on this date
        :param end_date: datetime
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
            daily_acc_hist.to_df(
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

    def interest_rate_breakdown(self, start_date: datetime, end_date: datetime, agg: bool = False,
                                filters = None, as_ayr: bool = True, as_prcnt: bool = True) -> pd.DataFrame:
        """
        Get the average interest rate for each account, or alternatively over each category specified by the
        aggregation level.

        :param start_date: datetime
            Start the daily summary on this date
        :param end_date: datetime
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
