import logging
from typing import Dict, Union, Iterable, Optional, List, Type, Any
import glob
import os

import pandas as pd

from .config import Config
from .utils import compare_pd_df

log = logging.getLogger("datasets")
field_names = Config.FieldNames


class DataField:
    DATE_TYPE = "date"

    def __init__(self, name: str, dtype: Union[str, type], mandatory: bool = True, def_value: Any = None):
        self.name = name
        self.dtype = dtype
        self.mandatory = mandatory
        self.def_value = def_value


class DataSource:
    """
    A base class that can used for any moneytrack data set.
    """

    # Specify a list of DataField that might be in the DataSource and their type. If "date" is
    # chosen as the type, the pd.to_datetime() function will be used to convert the input.
    # If a DataField is specified as mandatory, it must be in the DataSource, or an exception is thrown.
    fields: List[DataField] = []

    # List of field names that are NOT allowed in the DataSource
    forbidden_field_names: List[str] = []

    def __init__(self, df: pd.DataFrame):
        log.debug("Creating new {} object".format(self.__class__.__name__))
        self.df = self.validate_df(df)

    @classmethod
    def get_mandatory_field_names(cls) -> List[str]:
        """
        Returns a list of column names that are mandatory in the DataSource
        """
        return [f.name.upper() for f in cls.fields if f.mandatory is True]

    @classmethod
    def get_forbidden_field_names(cls):
        return [c.upper() for c in cls.forbidden_field_names]

    @classmethod
    def get_field_dtypes(cls) -> Dict[str, Union[type, str]]:
        return {f.name.upper(): f.dtype for f in cls.fields}

    @classmethod
    def has_mandatory_cols(cls, df: pd.DataFrame) -> bool:
        mandatory_cols = set(cls.get_mandatory_field_names())
        cols = set(df.columns)
        return len(mandatory_cols - cols) == 0

    @classmethod
    def has_forbidden_cols(cls, df: pd.DataFrame) -> bool:
        forbidden_cols = set(cls.get_forbidden_field_names())
        cols = set(df.columns)
        return len(forbidden_cols & cols) > 0

    @classmethod
    def coerce_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        for col, col_type in cls.get_field_dtypes().items():
            if col in df.columns:
                try:
                    if col_type == DataField.DATE_TYPE:
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
            cls.__name__, cls.get_mandatory_field_names(), df.columns
        )

        # Check that all the banned columns are not there
        assert not cls.has_forbidden_cols(df), "The DataFrame passed to {} should NOT have columns {}, but has {}".format(
            cls.__name__, cls.get_forbidden_field_names(), df.columns
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
        return {
            col_name: (dtype if dtype != DataField.DATE_TYPE else str)
            for col_name, dtype in cls.get_field_dtypes().items()
        }

    @classmethod
    def read_csv(cls, filename: str, **kwargs) -> pd.DataFrame:
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
    def from_csv_dir(cls, dir: str, file_ext = "csv") -> "DataSource":
        """
        Load DataSource from a csv file in a given directory. Any file in the directory with a matching file
        extension is checked for compatibility with the DataSource i.e. does it have any mandatory / forbidden
        columns, and are the datatypes compatible.

        :param dir: str
            Path to directory containing csv file/s
        :param file_ext: str
            Only check files with given extension
        :return: DataSource
        """

        files = glob.glob(os.path.join(dir, '*.' + file_ext.strip(".")))
        for file in files:
            try:
                return cls.from_csv(file)
            except (IOError, AssertionError):
                pass

        error_msg = """
        Could not find a valid csv file in {} for {}. The following files were checked {}
        """.format(dir, cls.__name__, files)

        raise IOError(error_msg)

    @classmethod
    def from_excel(cls, filename: str, sheet: Optional[str] = None):
        """
        Create an instance of class from csv file.
        """

        # If the sheet name is provided, try to load directly
        if sheet is not None:
            log.info("Loading {} from {} sheet {}".format(cls.__name__, filename, sheet))
            return cls(cls.read_excel(filename, sheet))

        # If the sheet isn't provided, try each sheet
        xl = pd.ExcelFile(filename)
        for sheet in xl.sheet_names:
            try:
                return cls(cls.read_excel(filename, sheet))
            except (IOError, AssertionError):
                pass

        raise IOError(
            "Could not find a sheet within {} that matches the expected format of ".format(filename, cls.__name__)
        )

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


class Accounts(DataSource):
    """
    A list of accounts, identified by a unique account key.
    """

    # The data types for each column
    fields = [
        DataField(name=field_names.ACCOUNT_KEY, dtype=str, mandatory=True),
        DataField(name=field_names.ACCOUNT_NBR, dtype=str, mandatory=False),
        DataField(name=field_names.SORT_CODE, dtype=str, mandatory=False),
        DataField(name=field_names.COMPANY, dtype=str, mandatory=False),
        DataField(name=field_names.ACCOUNT_TYP, dtype=str, mandatory=False),
        DataField(name=field_names.ISA, dtype=bool, mandatory=False),
        DataField(name=field_names.DESCRIPTION, dtype=str, mandatory=False),
    ]

    # Should not find these columns
    forbidden_field_names = [
        field_names.DATE,
        field_names.BALANCE,
        field_names.TRANSFER,
        field_names.TO_ACCOUNT_KEY,
        field_names.FROM_ACCOUNT_KEY,
    ]

    def __init__(self, df: pd.DataFrame):
        super(Accounts, self).__init__(df)

    def get_account_keys(self, without_ext: bool = True) -> List[str]:
        """
        Get a list of all account keys.

        :param without_ext: bool
            Do not return the EXT (external) account, used to track payments into / out of savings
        :return: List[str]
        """
        account_keys = self.df[field_names.ACCOUNT_KEY].values.tolist()
        if without_ext:
            account_keys = list(set(account_keys) - {Config.EXTERNAL_ACCOUNT_KEY})
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

        for column, values in filters.items():
            if column.upper() not in df_acc.columns:
                raise KeyError("The column {} is not in the Accounts data")
            if isinstance(values, str):
                values = [values]
            else:
                try:
                    iter(values)
                except TypeError:
                    values = [values]

            if isinstance(values[0], str):
                values = [str(v).upper() for v in values]
                df_acc = df_acc[df_acc[column.upper()].str.upper().isin(values)]
            else:
                df_acc = df_acc[df_acc[column.upper()].isin(values)]

        return df_acc[field_names.ACCOUNT_KEY].values.tolist()


class BalanceUpdates(DataSource):
    """
    An update of an account balance on a specified date. Any change in balance that is not due to a record in
    BalanceTransfers is assumed to be from interest.
    """

    # The data types for each column
    fields = [
        DataField(name=field_names.DATE, dtype=DataField.DATE_TYPE, mandatory=True),
        DataField(name=field_names.ACCOUNT_KEY, dtype=str, mandatory=True),
        DataField(name=field_names.BALANCE, dtype=float, mandatory=True),
        DataField(name=field_names.DESCRIPTION, dtype=str, mandatory=False),
    ]

    forbidden_field_names = [field_names.FROM_ACCOUNT_KEY, field_names.TO_ACCOUNT_KEY]

    def __init__(self, df):
        super(BalanceUpdates, self).__init__(df)

    def get_df_filtered(self, account_keys: List[str]) -> pd.DataFrame:
        df = self.get_df()
        mask = df[field_names.ACCOUNT_KEY].isin(account_keys)
        return df[mask]

    def get_acc_updates(self, account_key: str) -> pd.Series:
        """
        Get a Pandas series containing updates for a single account, containing update balance,
        and indexed by update date. Result is sorted ascending in date

        :param account_key: Unique key identifying an account
        :return: Series of account updates, indexed by datetime
        """
        df = self.get_df()
        df_acc = df[df[field_names.ACCOUNT_KEY] == account_key][[field_names.DATE, field_names.BALANCE]].copy()
        df_acc.set_index(field_names.DATE, inplace=True)
        df_acc.sort_index(inplace=True)

        return df_acc[field_names.BALANCE]


class BalanceTransfers(DataSource):
    """
    A historical record of all transfers made between accounts.
    """

    # The data types for each column
    fields = [
        DataField(name=field_names.DATE, dtype=DataField.DATE_TYPE, mandatory=True),
        DataField(name=field_names.AMOUNT, dtype=float, mandatory=True),
        DataField(name=field_names.FROM_ACCOUNT_KEY, dtype=str, mandatory=True),
        DataField(name=field_names.FROM_ACCOUNT_KEY, dtype=str, mandatory=True),
        DataField(name=field_names.DESCRIPTION, dtype=str, mandatory=False),
    ]

    forbidden_field_names = [field_names.ACCOUNT_KEY]

    def __init__(self, df):
        super(BalanceTransfers, self).__init__(df)

    def get_df_filtered(self, account_keys: List[str]) -> pd.DataFrame:
        df = self.get_df()
        mask = df[field_names.FROM_ACCOUNT_KEY].isin(account_keys) | df[field_names.TO_ACCOUNT_KEY].isin(account_keys)
        return df[mask]

    def _get_acc_transfers_to(self, account_key: str) -> pd.DataFrame:
        """
        Get a DataFrame containing balance transfers *to* an account, containing transfer amount,
        and transfer date ("amount" and "date" respectively).

        :param account_key: Unique key identifying an account
        :return: DataFrame of account transfers
        """
        df = self.get_df()
        return df[df[field_names.TO_ACCOUNT_KEY] == account_key][[field_names.DATE, field_names.AMOUNT]].copy()

    def _get_acc_transfers_from(self, account_key: str, signed: bool = False) -> pd.DataFrame:
        """
        Get a DataFrame containing balance transfers *from* an account, containing transfer amount,
        and transfer date ("amount" and "date" respectively).

        :param account_key: Unique key identifying an account
        :param signed: Should the transfer amounts be negated to indicate that they are outgoing payments
        :return: DataFrame of account transfers
        """
        df = self.get_df()
        df_t = df[df[field_names.FROM_ACCOUNT_KEY] == account_key][[field_names.DATE, field_names.AMOUNT]].copy()
        if signed:
            df_t[field_names.AMOUNT] = -df_t[field_names.AMOUNT]
        return df_t

    def get_acc_transfers(self, account_key) -> pd.Series:
        """
        Get a Series containing balance transfers to and from an account, containing transfer amount,
        and indexed by the transfer date. The sign of the amount indicates transfers to/from the account.

        :param account_key: Unique key identifying an account
        :return: DataFrame of account transfers
        """
        df = pd.concat([
            self._get_acc_transfers_to(account_key),
            self._get_acc_transfers_from(account_key, signed=True)
        ])
        series = df.groupby(field_names.DATE)[field_names.AMOUNT].sum()
        series.sort_index(inplace=True)
        series.rename(field_names.TRANSFER, inplace=True)
        return series
