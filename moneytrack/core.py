import glob
import logging
import os
from typing import Optional, Union, List, Dict

from .config import Config
from .datasets import Accounts, BalanceUpdates, BalanceTransfers
from .moneyframe import MoneyFrame
from .moneyframecollection import MoneyFrameCollection
from .utils import assert_type

log = logging.getLogger("core")
config = Config
field_names = config.FieldNames


class MoneyData(MoneyFrameCollection):
    """
    Container for all information about account portfolio:
        - Accounts: Details of each account in portfolio
        - Balance Updates: A checkpoint of an account balance at a point in time
        - Balance Transfers: Transfers between accounts, or from external sources
    """

    def __init__(self, accounts: Accounts, mf_collection: MoneyFrameCollection):
        """
        Create a MoneyData instance.

        :param accounts: Accounts
            Dataset containing details of each account in portfolio
        :param mf_collection: MoneyFrameCollection
            A MoneyFrameCollection which contains a money frame for each account
        """

        assert_type(accounts, Accounts)
        assert_type(mf_collection, MoneyFrameCollection)
        assert mf_collection.key_title == field_names.ACCOUNT_KEY, \
            "The keys of the MoneyFrameCollection should be account keys"

        super(MoneyData, self).__init__(mf_collection.moneyframes, mf_collection.key_title)
        self.accounts = accounts

    def filter_accounts(self, filters: Optional[Dict[str, Union[str, List[str]]]]) -> "MoneyData":
        """
        Filter MoneyData based on account properties

        :param filters: Optional[Dict[str, Union[str, List[str]]]]
            Optionally provide filters to be applied e.g.
                - {field_names.ACCOUNT_KEY: "TSB"}
                - {field_names.ACCOUNT_KEY: ["TSB", "LLOYDS"], field_names.ACCOUNT_TYPE: "CURRENT"}
        :return: MoneyData
            A filtered copy of MoneyData
        """
        if filters is None:
            return self

        acc_keys = self.accounts.get_matching_account_keys(filters, without_ext=True)
        mf_collection = self.filter(lambda x: x in acc_keys)
        return MoneyData(accounts=self.accounts, mf_collection=mf_collection)

    def groupby_accounts(self, by: Union[bool, str] = True) -> MoneyFrameCollection:
        """
        Aggregate the dictionary containing account histories. Can be aggregated by any column in the Accounts data
        e.g. passing agg=field_names.ACCOUNT_TYPE

        Alternatively False can be passed to do no aggregation (i.e. aggregated by field_names.ACCOUNT_KEY), or True
        can be passed to aggregate ALL accounts.

        :param by: Union[bool, str]
            How to aggregate the data e.g.
                - field_names.ACCOUNT_TYPE: will aggregate by account type
                - False: will perform no aggregation
                - True: will aggregate all the accounts
        :return: MoneyFrameCollection
        """

        if by is True:
            return self.groupby(lambda x: "ALL", key_title="ALL")
        if by is False:
            return self

        assert_type(by, str)
        by = by.upper()

        if by == field_names.ACCOUNT_KEY.upper():
            return self

        df_acc = self.accounts.get_df().copy()
        # Make all the column names upper case
        df_acc.columns = [c.upper() for c in df_acc.columns]

        # Create dictionary that maps each account key to group
        keys = df_acc[field_names.ACCOUNT_KEY].str.upper().values
        values = df_acc[by].values

        d = dict(zip(keys, values))

        # Perform groupby and return result
        return self.groupby(lambda x: d.get(x.upper(), None), key_title=by)

    @classmethod
    def from_updates(cls, accounts: Accounts, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers):
        """
        Create a MoneyData instance.

        :param accounts: Accounts
            Dataset containing details of each account in portfolio
        :param balance_updates: BalanceUpdates
            Dataset containing a history of account balance checkpoints e.g. what was the balance on a given day
        :param balance_transfers: BalanceTransfers
            Dataset containing a history of transfers between accounts
        """

        # Create a daily account history for each account
        mf_collection = MoneyFrameCollection({
            acc_key: MoneyFrame.from_updates_and_transfers(
                balance_updates,
                balance_transfers,
                acc_key
            )
            for acc_key in accounts.get_account_keys()
        }, field_names.ACCOUNT_KEY)

        return cls(accounts, mf_collection)

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
        return MoneyData.from_updates(
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
        return MoneyData.from_updates(
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

    def __getitem__(self, item):
        return MoneyData(
            self.accounts,
            super(MoneyData, self).__getitem__(item)
        )
