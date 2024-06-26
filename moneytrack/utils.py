import logging
from enum import Enum
from typing import Union, Iterable, TypeVar, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar

from .exceptions import NoSolutionFoundError

log = logging.getLogger("utils")

TNumeric = TypeVar('TNumeric', int, float)

class DateRanges:

    @classmethod
    def tax_year(cls, year_starting: Union[str, int]):
        year_starting = int(year_starting)
        return slice(datetime(year_starting, 4, 6), datetime(year_starting+1, 4, 5))

    @classmethod
    def calendar_year(cls, year: Union[str, int]):
        year = int(year)
        return slice(datetime(year, 1, 1), datetime(year, 12, 31))

    @classmethod
    def from_date(cls, year: Union[str, int], month: Union[str, int] = 1, day: Union[str, int] = 1):
        return slice(datetime(year, month, day), None)


class SparseVector:
    """
    A sparse representation of a 1D vector. Behind the scenes it uses a scipy.sparse.csr_matrix, but
    this wrapper has a much simplified interface.
    """

    def __init__(self, values: Iterable[TNumeric], indices: Iterable[int], size: Optional[int] = None):
        """
        Create a sparse vector with values, and the indices at which they occur. All other elements
        are zero. As an example, the vector [0, 0, 4.3, 0, 3.4, 0], could be created as:

            SparseVector([4.3, 3.4], [2, 4], 6)

        :param values: Iterable[TNumeric]
            The non-zero values of the vector e.g. [4.3, 3.4]
        :param indices: Iterable[int]
            The indices at which the values occur e.g. [2, 4]
        :param size: Optional[int]
            Optionally provide the length of the array e.g. 6
        """
        if size is not None and np.max(indices) >= size:
            raise ValueError("The size parameter is smaller or equal to an element of indices")

        try:
            shape = (1, size) if size is not None else None
            self.sparse_mat = scipy.sparse.csr_matrix(
                (values, indices, [0, len(values)]), shape=shape
            )
        except ValueError as e:
            raise ValueError("Could not parse inputs values = {}, indices = {}, size = {}."
                             "Full error: {}".format(values, indices, size, repr(e)))

    def dense(self) -> np.array:
        """
        Obtain the dense representation of the sparse vector
        :return: np.array
        """
        return self.sparse_mat.toarray()[0]

    @staticmethod
    def from_dense(self, ar: Iterable[TNumeric]) -> "SparseVector":
        """
        Create a sparse representation of a dense vector

        :param self:
        :param ar:
        :return:
        """
        raise NotImplementedError()

def calc_real_pos_roots(p: Iterable[float]):
    """
    Find the real positive roots of a polynomial:

        p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    :param p: polynomial coefficients
    :return: real positive roots
    """
    roots = np.roots(p)
    real_roots = roots[~np.iscomplex(roots)].real
    real_pos_roots = real_roots[real_roots > 0]
    return real_pos_roots


def create_daily_transfer_record(trans_days: Iterable[int], trans_amts: Iterable[TNumeric]) -> np.array:
    """
    Take an array of transfers, and the day on which they happened, and
    create an array, where each element represents a day, and the value
    represents the total transfer amount e.g. trans_days = [2, 5],
    trans_amts = [0.5, 1.4] becomes [0.0, 0.0, 0.5, 0.0, 0.0, 1.4]

    :param trans_days: array[int]
        The day on which the transfer was made
    :param trans_amts: array[double]
        The amount of the transfer
    :return: daily transfer record
    """
    log.debug("creating daily transfer record using days {} and amounts {}".format(str(trans_days), str(trans_amts)))
    return SparseVector(values=trans_amts, indices=trans_days).dense()


def calc_avg_interest_rate(start_bal, end_bal, num_days, trans_days, trans_amts, method="AUTO"):
    """
    sum amt_i * (1 + ir) ^ ndays_i = end_bal

    :param start_bal: double
        The starting balance on day 0
    :param end_bal: double
        The finishing balance on day "ndays"
    :param num_days: int
        The number of days between the start_bal and end_bal
    :param trans_days: List[int]
        A list of days where transfers were made too / from the account
    :param trans_amts: List[double]
        A list of amounts for transfers that were made too / from the account
    :param method: str
        Should the numerical or analytic method be used.
    :return: The average daily interest rate over over the period

    :raises KeyError: Incorrect input was given
    :raises NoSolutionFoundError: Cannot find a solution (an interest rate)

    """

    # Can't calculate an interest rate over zero days
    if num_days == 0:
        return 0.0

    trans_days = np.asarray(trans_days)
    trans_amts = np.asarray(trans_amts)

    trans_days_in = [0, num_days] + trans_days.tolist()
    trans_amts_in = [start_bal, -end_bal] + trans_amts.tolist()

    rec = create_daily_transfer_record(trans_days_in, trans_amts_in)
    # If every element is zero, the account has always been empty.
    if not rec.any():
        return 0.0

    if (method.upper() == "AUTO" and num_days > 100.0) or method.upper() == "NUMERICAL":
        method = "NUMERICAL"
    elif (method.upper() == "AUTO" and num_days <= 100.0) or method.upper() == "ANALYTIC":
        method = "ANALYTICAL"
    else:
        raise KeyError("Method must be in [AUTO, NUMERICAL, ANALYTICAL]. {} was given.".format(method))

    # Determine the average interest rate
    if method == "NUMERICAL":
        # This polynomial should only have one real positive root that we're interest in.
        # It's therefore not necessary to find all of the roots of the polynomial.
        # When the order of the polynomial becomes large, it's much faster to just solve it numerically.
        # Note that num_days > 100 hasn't really been tuned for performance.

        def f_target(x): return np.power(np.polyval(rec, x), 2.0)

        # Try using a few sets of bracketing intervals
        bracketing_intervals = [
            (0.0, 1.0, 1.1),
            (1.0 - 1.0e-1, 1.0, 1.1),
            (1.0 - 1.0e-3, 1.0, 1.1),
            (1.0 - 1.0e-6, 1.0, 1.1),
            (1.0 - 1.0e-12, 1.0, 1.1),
        ]

        result = None
        for bracket in bracketing_intervals:
            try:
                result = minimize_scalar(f_target, bracket=bracket, method="brent")
            except ValueError:
                pass
            if result is not None and result.success:
                break

        if result is None or not result.success:
            debug_log = "start_bal={}, end_bal={}, num_days={}, trans_days={}, trans_amts={}".format(
                start_bal, end_bal, num_days, trans_days, trans_amts,
            )
            raise NoSolutionFoundError(
                "Could not find a numerical solution with the following inputs:\n" + debug_log
            )
        return result.x - 1.0

    if method == "ANALYTICAL":
        real_pos_roots = calc_real_pos_roots(rec)
        if len(real_pos_roots) == 0:
            debug_log = "start_bal={}, end_bal={}, num_days={}, trans_days={}, trans_amts={}".format(
                start_bal, end_bal, num_days, trans_days, trans_amts,
            )
            raise NoSolutionFoundError(
                "Could not find any real analytic solutions with the following inputs:\n" + debug_log
            )
        return real_pos_roots[0] - 1.0


def calc_daily_balances(start_bal: float, end_day: int, daily_rate: float, start_day: int = 0):
    """
    Calculate the balance on each day for a given starting balance and interest rate.

    :param start_bal: float
        The starting balance (on start_day)
    :param end_day: int
        Number of days
    :param daily_rate:
        The daily interest rate
    :param start_day:
        The day on which the balance was paid in i.e. on all days preceding the balance was zero
    :return: np.array[float]
        The balance on each day.
    """
    return np.concatenate([np.zeros(start_day), start_bal * np.power(daily_rate + 1.0, np.arange(end_day - start_day))])


def calc_daily_balances_w_transfers(start_bal, num_days, daily_rate, trans_amts=None, trans_days=None):
    """
    Calculate the balance on each day for a given starting balance and interest rate. Also the option
    to pass transfers that are paid into / out of the account during the time frame.

    :param start_bal: float
        The starting balance on day zero
    :param num_days: int
        Number of days
    :param daily_rate: float
        The daily interest rate
    :param trans_amts: list[float]
        A list of amounts for transfers that were made too / from the account
    :param trans_days: list[int]
        A list of days where transfers were made too / from the account
    :return: list[double]
        The balance on each day.
    """
    if trans_amts is None:
        trans_amts = []
    daily_bals = calc_daily_balances(start_bal, num_days, daily_rate)
    if trans_amts is not None and trans_days is not None:
        for day, amt in zip(trans_days, trans_amts):
            daily_bals += calc_daily_balances(start_bal=amt, start_day=day, end_day=num_days, daily_rate=daily_rate)
    return daily_bals


def cross_join(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    df_a["dummy_"] = 1
    df_b["dummy_"] = 1
    dj_j = df_a.merge(df_b, on="dummy_", how="inner")
    df_a.drop("dummy_", axis=1, inplace=True)
    df_b.drop("dummy_", axis=1, inplace=True)
    return dj_j.drop("dummy_", axis=1)


def dates_between(start_date, end_date):
    """
    Get a list of all dates between a start and end date.

    :param start_date:
    :param end_date:
    :return:
    """
    num_days = (end_date - start_date).days
    return [start_date + pd.Timedelta(days=i) for i in range(num_days + 1)]


def compare_pd_df(df_a: pd.DataFrame, df_b: pd.DataFrame, sort: bool = True) -> bool:
    """
    Compare two pandas DataFrame objects, and return boolean.
    """
    if set(df_a.columns.values.tolist()) != set(df_b.columns.values.tolist()):
        return False
    cols = df_a.columns.values.tolist()

    if sort:
        df_a_comp = df_a[cols].sort_values(by=cols).reset_index(drop=True)
        df_b_comp = df_b[cols].sort_values(by=cols).reset_index(drop=True)
        return df_a_comp.equals(df_b_comp)

    df_a_comp = df_a[cols].reset_index(drop=True)
    df_b_comp = df_b[cols].reset_index(drop=True)
    return df_a_comp.equals(df_b_comp)


def adr_to_ayr(adr: Union[np.array, float]) -> Union[np.array, float]:
    """
    Convert the average daily interest rate, to an average yearly rate
    :param adr: Union[np.array, float]
        The average daily interest rate
    :return: Union[np.array, float]
        The average yearly interest rate
    """
    return np.power(1.0 + adr, 365.0) - 1.0


def ayr_to_adr(ayr: Union[np.array, float]) -> Union[np.array, float]:
    """
    Convert the average daily interest rate, to an average yearly rate
    :param ayr: Union[np.array, float]
        The average yearly interest rate
    :return: Union[np.array, float]
        The average daily interest rate
    """
    return np.power(1.0 + ayr, 1.0 / 365.0) - 1.0


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> Optional[T]:
    """
    Return the first argument that is not None
    :param args: Optional[T]
    :return: Optional[T]
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


class RangeOverlap(Enum):
    BELOW_REF = 1
    ABOVE_REF = 2
    EQUAL = 3
    WITHIN_REF = 4
    CONTAINS_REF = 5
    BELOW_REF_OVERLAP = 6
    ABOVE_REF_OVERLAP = 7


def get_range_overlap_cat(reference_range: Iterable[float], comparison_range: Iterable[float]):
    """
    Compare one range to another.
    :param reference_range: Iterable[float]
        A reference range. The result will be wrt this i.e. BELOW would indicate that the comparison range is BELOW
        the reference range.
    :param comparison_range: Iterable[float]
        Comparison range.
    :return: RangeOverlap
    """
    try:
        ref_low, ref_high = reference_range
        cmp_low, cmp_high = comparison_range
    except (TypeError, ValueError):
        raise ValueError("Could not unpack {} and {} into ranges".format(reference_range, comparison_range))

    assert ref_low <= ref_high, "Lower bound of range should be smaller than upper: {}".format(reference_range)
    assert cmp_low <= cmp_high, "Lower bound of range should be smaller than upper: {}".format(comparison_range)

    if cmp_high <= ref_low:
        return RangeOverlap.BELOW_REF
    if cmp_low >= ref_high:
        return RangeOverlap.ABOVE_REF
    if (cmp_low == ref_low) and (cmp_high == ref_high):
        return RangeOverlap.EQUAL
    if (cmp_low >= ref_low) and (cmp_high <= ref_high):
        return RangeOverlap.WITHIN_REF
    if (cmp_low <= ref_low) and (cmp_high >= ref_high):
        return RangeOverlap.CONTAINS_REF
    if (cmp_low < ref_low) and (cmp_high >= ref_high):
        return RangeOverlap.BELOW_REF_OVERLAP
    if (cmp_high > ref_high) and (cmp_high >= ref_high):
        return RangeOverlap.ABOVE_REF_OVERLAP

    raise ValueError("This range overlap is not defined {} {}".format(reference_range, comparison_range))


def get_range_overlap(range_1, range_2):
    reference_range, comparison_range = range_1, range_2

    overlap_type = get_range_overlap_cat(reference_range=reference_range, comparison_range=comparison_range)

    if overlap_type == RangeOverlap.EQUAL:
        return reference_range
    if overlap_type == RangeOverlap.CONTAINS_REF:
        return reference_range
    if overlap_type == RangeOverlap.WITHIN_REF:
        return comparison_range
    if overlap_type in (RangeOverlap.ABOVE_REF, RangeOverlap.BELOW_REF):
        return None
    if overlap_type == RangeOverlap.BELOW_REF_OVERLAP:
        return reference_range[0], comparison_range[1]
    if overlap_type == RangeOverlap.ABOVE_REF_OVERLAP:
        return comparison_range[0], reference_range[1]


def assert_type(obj, typ, optional=False):
    if optional and obj is None:
        return
    if not isinstance(obj, typ):
        raise TypeError("Expected type {} but received {}".format(typ.__name__, obj.__class__.__name__))
