from collections import defaultdict
from datetime import datetime
from typing import Dict, Hashable, Optional, Callable, Union, List, Tuple, Any

import pandas as pd

from .config import Config
from .moneyframe import MoneyFrame
from .utils import coalesce

field_names = Config.FieldNames


class MoneyFrameCollection:

    def __init__(self, moneyframes: Dict[Hashable, MoneyFrame], key_title=Optional[str]):
        """
        A MoneyFrameCollection object is a collection of moneyframes, indexed by a key i.e. a dictionary
        of MoneyFrames.

        :param moneyframes: Dict[Hashable, MoneyFrame]
            A dictionary of MoneyFrames
        :param key_title: Optional[str]
            What does the key represent e.g. "account_key"
        """
        self.key_title = key_title
        self.moneyframes = moneyframes

    def filter(self, function: Callable[[Hashable], bool]) -> "MoneyFrameCollection":
        """
        Filter the MoneyFrames for any matching keys

        :param function: Callable[[Hashable], bool]
            Function applied to each key which returns a boolean to apply filter
        :return: MoneyFrameCollection
        """
        return MoneyFrameCollection(
            moneyframes={k: v for k, v in self.moneyframes.items() if function(k)},
            key_title=self.key_title
        )

    def __len__(self):
        return len(self.moneyframes)

    def sum(self) -> MoneyFrame:
        """
        Aggregate all MoneyFrames into a single MoneyFrame
        :return: MoneyFrame
        """
        if len(self) == 0:
            return MoneyFrame.create_empty()
        return MoneyFrame.from_sum(self.moneyframes.values())

    def groupby(self,
                by: Union[Callable[[Hashable], Hashable], Dict[Hashable, Hashable], List[Tuple[Hashable, Hashable]],
                          Tuple[List[Hashable], List[Hashable]]]
                , key_title: Optional[str] = None) -> "MoneyFrameCollection":
        """
        Aggregate with a new set of keys, using a mapping from the old keys to the new keys.

        :param by:
            Used to determine the groups for the groupby. If by is a function, it’s called on each key.
            If a dict is passed, the dict VALUES will be used to determine the groups (the KEYS are first aligned)
        :param key_title: Optional[str]
            Title for the new keys
        :return: MoneyFrameCollection
        """

        # Try to manipulate into
        try:
            by = dict(by)
        except (TypeError, ValueError):
            try:
                by = dict(zip(*by))
            except (TypeError, ValueError, SyntaxError):
                pass

        d = defaultdict(MoneyFrame.create_empty)
        for k, mf in self.moneyframes.items():
            new_key = by.get(k, None) if isinstance(by, dict) else by(k)

            if new_key is not None:
                d[new_key] += mf

        return MoneyFrameCollection(d, key_title=key_title)

    def map_values(self, function: Callable[[MoneyFrame], Any], as_list: bool = False) \
            -> Union[Dict[Hashable, Any], List[Any]]:

        if as_list:
            return [function(mf) for k, mf in self.moneyframes.items()]
        return {k: function(mf) for k, mf in self.moneyframes.items()}

    def avg_interest_rates(self, as_df=False, **kwargs) -> Union[Dict[Hashable, float], pd.DataFrame]:
        """
        Get the average interest rate for each account, or alternatively over each category specified by the
        aggregation level.

        :param as_df: bool
            Should the result be returned as a pandas DataFrame or a dictionary
        :param start_date: datetime
            Start the daily summary on this date
        :param end_date: datetime
            End the daily summary on this date
        :param as_ayr: bool
            When True, return the average yearly rate, rather than the average daily rate
        :param as_prcnt: bool
            When True, return as a percentage rather than a fraction
        :return: Union[Dict[Hashable, float], pd.DataFrame]
            DataFrame containing the interest rate breakdown over the period specified
        """

        d = self.map_values(lambda x: x.calc_avg_interest_rate(**kwargs))
        if as_df:
            df = pd.DataFrame(index=d.keys(), data=d.values(), columns=[field_names.INTEREST_RATE])
            if self.key_title is not None:
                df.index.name = self.key_title
            return df.sort_values(field_names.INTEREST_RATE)
        return d

    def __getitem__(self, x) -> Union["MoneyFrameCollection", MoneyFrame]:

        # First check to see if x is one of the MoneyFrameCollection keys.
        try:
            if x in self.keys():
                return self.moneyframes[x]
        except TypeError:
            # If x is not hashable
            pass

        # If x is not one of the MoneyFrameCollection keys, apply the filter to each MoneyFrame
        # in the collection.
        return MoneyFrameCollection(
            {k: v.__getitem__(x) for k, v in self.items()},
            self.key_title
        )

    def items(self):
        return self.moneyframes.items()

    def values(self):
        return self.moneyframes.values()

    def keys(self):
        return self.moneyframes.keys()

    def to_df(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
              inc_cum_interest_rate: bool = False, inc_interest_rate: bool = False,
              as_ayr: bool = True, as_prcnt: bool = True, **kwargs) -> pd.DataFrame:
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
        """
        key_title = coalesce(self.key_title, "AGG_KEY")
        dfs = [
            mf.to_df(
                start_date=start_date,
                end_date=end_date,
                inc_cum_interest_rate=inc_cum_interest_rate,
                inc_interest_rate=inc_interest_rate,
                as_ayr=as_ayr,
                as_prcnt=as_prcnt,
                **kwargs
            ).assign(**{key_title: key})
            for key, mf in self.moneyframes.items()
        ]

        return pd.concat(dfs).reset_index().set_index([field_names.DATE, key_title])
