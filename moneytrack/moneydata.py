import logging
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
        - MoneyFrameCollection: Contains daily updates of balances, interest payments, and transfers for each account
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
            "For a MoneyData object, the keys of the MoneyFrameCollection should be account keys. Not {}".format(
                mf_collection.key_title)

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
    def from_updates_and_transfers(cls, accounts: Accounts, balance_updates: BalanceUpdates, balance_transfers: BalanceTransfers):
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
        return MoneyData.from_updates_and_transfers(
            Accounts.from_csv(accounts_path),
            BalanceUpdates.from_csv(balance_updates_path),
            BalanceTransfers.from_csv(balance_transfers_path),
        )

    @classmethod
    def from_excel(cls, path: str, accounts_sheet=None, transfers_sheet=None, updates_sheet=None) -> "MoneyData":
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
        return MoneyData.from_updates_and_transfers(
            Accounts.from_excel(path, accounts_sheet),
            BalanceUpdates.from_excel(path, updates_sheet),
            BalanceTransfers.from_excel(path, transfers_sheet),
        )

    @classmethod
    def from_csv_dir(cls, path: str, file_ext: str = "csv") -> "MoneyData":
        """
        Helper method to create a MoneyData instance from a directory containing csv files.

        :param path: str
            Directory containing csv files
        :param file_ext: str
            File extension of the csv files
        :return: MoneyData
        """
        return cls.from_updates_and_transfers(
            accounts=Accounts.from_csv_dir(path, file_ext),
            balance_updates=BalanceUpdates.from_csv_dir(path, file_ext),
            balance_transfers=BalanceTransfers.from_csv_dir(path, file_ext),
        )

    def __getitem__(self, item) -> Union["MoneyData", MoneyFrame]:

        mf = super(MoneyData, self).__getitem__(item)
        if isinstance(mf, MoneyFrameCollection):
            return MoneyData(self.accounts, mf)

        return mf
