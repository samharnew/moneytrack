import pandas as pd
import logging

from collections import OrderedDict
from typing import Dict, Union, Iterable, Optional, List, Type

from .utils import compare_pd_df
from .config import Config

log = logging.getLogger("datasets")


class DataSource:
    """
    A base class that can used for any moneytrack data set.
    """

    # Specify a dict of columns that *must* be in the DataSource and their type. If "date" is
    # chosen as the type, the pd.to_datetime() function will be used to convert the input.
    dtypes = None
    DATE_TYPE = "date"

    def __init__(self, df: pd.DataFrame):
        log.debug("Creating new {} object".format(self.__class__.__name__))
        self.df = self.validate_df(df)

    @classmethod
    def mandatory_cols(cls) -> List[str]:
        """
        Returns a list of column names that are mandatory in the DataSource
        """
        return list(cls.get_dtypes().keys())

    @classmethod
    def get_dtypes(cls):
        return {k.upper(): v for k, v in cls.dtypes.items()}

    @classmethod
    def has_mandatory_cols(cls, df: pd.DataFrame) -> bool:
        exp = set(cls.mandatory_cols())
        cols = set(df.columns)
        return len(exp - cols) == 0

    @classmethod
    def coerce_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        for col, col_type in cls.get_dtypes().items():
            try:
                if col_type == cls.DATE_TYPE:
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(col_type)
            except Exception:
                raise TypeError("Cannot cast column '{}' to dtype={}".format(col, col_type))
        return df

    @classmethod
    def validate_df(cls, df: pd.DataFrame) -> pd.DataFrame:

        # Make all columns upper case
        df.columns = map(str.upper, df.columns)

        # Check that all the required columns are there
        assert cls.has_mandatory_cols(df), "The DataFrame passed to {} should have columns {}, but has {}".format(
            cls.__name__, cls.mandatory_cols(), df.columns
        )

        # Cast all of the columns to their correct data types
        df = cls.coerce_dtypes(df)
        return df

    def get_df(self) -> pd.DataFrame:
        if self.df is None:
            raise NotImplementedError("Any subclass of DataSource must implement a self.df member")
        return self.df

    def to_csv(self, path):
        self.get_df().to_csv(path, index=False)

    @classmethod
    def get_dtypes_pandas(cls: "DataSource") -> Dict[str, Type]:
        return {colnm: (dtype if dtype != cls.DATE_TYPE else str) for colnm, dtype in cls.get_dtypes().items()}

    @classmethod
    def read_csv(cls: "DataSource", filename: str, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_csv(filename, dtype=cls.get_dtypes_pandas(), **kwargs)
        except (KeyError, ValueError):
            raise IOError("Could not load {} from csv file {}".format(cls.__name__, filename))
        return df

    @classmethod
    def read_excel(cls, filename: str, sheet: str, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_excel(filename, sheet_name=sheet, dtype=cls.get_dtypes_pandas(), **kwargs)
        except (KeyError, ValueError):
            raise IOError("Could not load {} from excel file {}, sheet {}".format(cls.__name__, filename, sheet))
        return df

    @classmethod
    def from_csv(cls, filename: str):
        """
        Create an instance of class from csv file.
        """
        log.info("Loading {} from {}".format(cls.__name__, filename))
        return cls(cls.read_csv(filename))

    @classmethod
    def from_excel(cls, filename: str, sheet: str):
        """
        Create an instance of class from csv file.
        """
        log.info("Loading {} from {} sheet {}".format(cls.__name__, filename, sheet))
        return cls(cls.read_excel(filename, sheet))

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create an instance of class from dictionary.
        """
        return cls(pd.DataFrame.from_dict(d))

    def equals(self, other: "DataSource") -> bool:
        return compare_pd_df(other.get_df(), self.get_df())

    def __str__(self):
        return self.__class__.__name__ + ":\n" + str(self.df)


class DataFields:
    # Core Dataset Fields
    ACCOUNT_KEY = "ACCOUNT_KEY"
    DATE = "DATE"
    BALANCE = "BALANCE"
    FROM_ACCOUNT_KEY = "FROM_ACCOUNT_KEY"
    TO_ACCOUNT_KEY = "TO_ACCOUNT_KEY"
    AMOUNT = "AMOUNT"

    # Daily Summary Fields
    INTEREST = "INTEREST"
    TRANSFER = "TRANSFER"

    # Others
    PREV_BALANCE = "PREV_BALANCE"
    PREV_DATE = "PREV_DATE"
    INTEREST_RATE = "INTEREST_RATE"
    CUM_INTEREST_RATE = "CUM_INTEREST_RATE"

    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    START_BALANCE = "START_BALANCE"
    END_BALANCE = "END_BALANCE"


class Accounts(DataSource):
    """
    A list of accounts, identified by a unique account key.
    """

    # The data types for each column
    dtypes = OrderedDict([
        (DataFields.ACCOUNT_KEY, str),
        ("ACCOUNT_NBR", str),
        ("SORT_CODE", str),
        ("COMPANY", str),
        ("ACCOUNT_TYP", str),
        ("ISA", bool),
    ])

    def __init__(self, df: pd.DataFrame):
        super(Accounts, self).__init__(df)

    def get_account_keys(self, without_ext: bool = True) -> List[str]:
        """
        Get a list of all account keys.

        :param without_ext: bool
            Do not return the EXT (external) account, used to track payments into / out of savings
        :return: List[str]
        """
        account_keys = self.df[DataFields.ACCOUNT_KEY].values.tolist()
        if without_ext:
            account_keys = list(set(account_keys) - {Config.external_account_key})
        return account_keys

    def get_matching_account_keys(self, filters: Optional[Dict[str, Union[str, Iterable[str]]]] = None,
                                  without_ext: bool = True) -> List[str]:
        """
        Get a list of all account keys that match a list of filters given.

        :param filters: None, Dict[str, str], Dict[str, Iterable[str]]
            - None: will apply no filters
            - Dict[str, str]: each key/value pair equates to a columnName/filterOnValue e.g. {"account_typ":"P2P"}
              will filter only the P2P (peer to peer) account type.
            - Dict[str, Iterable[str]]: each key/value pair equates to a columnName/filterOnValues e.g.
              {"account_typ"->["P2P", "StocksAndShares"]} will filter all accounts with account types of P2P or
              StocksAndShares
        :param without_ext: bool
            Do not return the EXT (external) account, used to track payments into / out of savings
        :return: List[str]
            A list of matching account keys

        :raises KeyError: If the account property you're filtering on does not exist
        """
        if filters is None:
            return self.get_account_keys(without_ext)

        df_acc = self.get_df().copy(deep=False)
        df_acc.columns = [c.upper() for c in df_acc.columns]

        for column, values in filters.items():
            if column.upper() not in df_acc.columns:
                raise KeyError("The column {} is not in the Accounts data")
            if isinstance(values, str):
                values = [values]
            values = [v.upper() for v in values]
            df_acc = df_acc[df_acc[column.upper()].str.upper().isin(values)]

        return df_acc[DataFields.ACCOUNT_KEY].values.tolist()


class BalanceUpdates(DataSource):
    """
    An update of an account balance on a specified date. Any change in balance that is not due to a record in
    BalanceTransfers is assumed to be from interest.
    """

    # The data types for each column
    dtypes = OrderedDict([
        (DataFields.ACCOUNT_KEY, str),
        (DataFields.BALANCE, float),
        (DataFields.DATE, DataSource.DATE_TYPE),
    ])

    def __init__(self, df):
        super(BalanceUpdates, self).__init__(df)

    def get_df_filtered(self, account_keys: List[str]) -> pd.DataFrame:
        df = self.get_df()
        mask = df[DataFields.ACCOUNT_KEY].isin(account_keys)
        return df[mask]

    def get_acc_updates(self, account_key: str, prev_update_cols: bool = False) -> pd.DataFrame:
        """
        Get a DataFrame containing updates for a single account, containing update balance,
        and update date ("balance" and "date" respectively). Result is sorted ascending in date

        :param account_key: Unique key identifying an account
        :param prev_update_cols: Add additional columns to the DataFrame with "prev_balance" and "prev_date"
        :return: DataFrame of account updates
        """
        df = self.get_df()
        df_acc = df[df[DataFields.ACCOUNT_KEY] == account_key][[DataFields.DATE, DataFields.BALANCE]].copy()
        df_acc.sort_values(DataFields.DATE, ascending=True, inplace=True)

        if prev_update_cols:
            df_acc[DataFields.PREV_BALANCE] = df_acc[DataFields.BALANCE].shift(1)
            df_acc[DataFields.PREV_DATE] = df_acc[DataFields.DATE].shift(1)

        return df_acc


class BalanceTransfers(DataSource):
    """
    A historical record of all transfers made between accounts.
    """

    # The data types for each column
    dtypes = OrderedDict([
        (DataFields.DATE, DataSource.DATE_TYPE),
        (DataFields.AMOUNT, float),
        (DataFields.FROM_ACCOUNT_KEY, str),
        (DataFields.TO_ACCOUNT_KEY, str),
    ])

    def __init__(self, df):
        super(BalanceTransfers, self).__init__(df)

    def get_df_filtered(self, account_keys: List[str]) -> pd.DataFrame:
        df = self.get_df()
        mask = df[DataFields.FROM_ACCOUNT_KEY].isin(account_keys) | df[DataFields.TO_ACCOUNT_KEY].isin(account_keys)
        return df[mask]

    def get_acc_transfers_to(self, account_key: str) -> pd.DataFrame:
        """
        Get a DataFrame containing balance transfers *to* an account, containing transfer amount,
        and transfer date ("amount" and "date" respectively).

        :param account_key: Unique key identifying an account
        :return: DataFrame of account transfers
        """
        df = self.get_df()
        return df[df[DataFields.TO_ACCOUNT_KEY] == account_key][[DataFields.DATE, DataFields.AMOUNT]].copy()

    def get_acc_transfers_from(self, account_key: str, signed: bool = False) -> pd.DataFrame:
        """
        Get a DataFrame containing balance transfers *from* an account, containing transfer amount,
        and transfer date ("amount" and "date" respectively).

        :param account_key: Unique key identifying an account
        :param signed: Should the transfer amounts be negated to indicate that they are outgoing payments
        :return: DataFrame of account transfers
        """
        df = self.get_df()
        df_t = df[df[DataFields.FROM_ACCOUNT_KEY] == account_key][[DataFields.DATE, DataFields.AMOUNT]].copy()
        if signed:
            df_t[DataFields.AMOUNT] = -df_t[DataFields.AMOUNT]
        return df_t

    def get_acc_transfers(self, account_key) -> pd.DataFrame:
        """
        Get a DataFrame containing balance transfers to and from an account, containing transfer amount,
        and transfer date ("amount" and "date" respectively). The sign of the amount indicates transfers to/from
        the account.

        :param account_key: Unique key identifying an account
        :return: DataFrame of account transfers
        """
        return pd.concat([
            self.get_acc_transfers_to(account_key),
            self.get_acc_transfers_from(account_key, signed=True)
        ])
