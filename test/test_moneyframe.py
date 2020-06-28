import unittest
import pandas as pd
import numpy as np
from moneytrack import BalanceTransfers, BalanceUpdates, MoneyFrame
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

class TestCore(unittest.TestCase):

    def test_get_daily_balances(self):
        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": [],
            "to_account_key": [],
            "amount": [],
            "date": [],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": ["1", "1", "1", "1", "1"],
            "balance": [100.0, 200.0, 220.0, 220.0, 200.0],
            "date": ["2019-01-01", "2019-01-03", "2019-01-04", "2019-01-06", "2019-01-08"],
        })

        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans,
                                                     balance_updates=balance_updates, account_key="1")
        daily_bals = hist.get_daily_balance()
        exp = pd.DataFrame({
            "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05", "2019-01-06",
                                    "2019-01-07", "2019-01-08"]),
            "balance": [100.0, 141.421356, 200.0, 220.0, 220.0, 220.0, 209.761770, 200.0],
        }).set_index("date")

        np.testing.assert_equal(exp.index.values, daily_bals.index.values)
        np.testing.assert_array_almost_equal(exp["balance"].values, daily_bals.values)

    def test_from_fixed_rate(self):

        dah = MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0)
        self.assertTrue(len(dah) == 5)
        self.assertTrue(dah.max_date() == datetime.today().date())

        dah = MoneyFrame.from_fixed_rate(days=("2020-01-01", "2020-01-05"), start_bal=100.0, ayr_prcnt=5.0)
        self.assertTrue(len(dah) == 5)

    def test_slice(self):

        dah = MoneyFrame.from_fixed_rate(days=("2020-01-01", "2020-01-05"), start_bal=100.0, ayr_prcnt=5.0)
        self.assertTrue(len(dah[2]) == 1)
        self.assertTrue(len(dah[2:3]) == 1)
        self.assertTrue(len(dah[2:4]) == 2)
        self.assertTrue(len(dah["2020-01-01"]) == 1)
        self.assertTrue(len(dah["2020-01-01":"2020-01-03"]) == 3)

    def test_get_daily_balances_with_trans(self):
        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": ["1", "2", "1", "2", "1", "2"],
            "to_account_key": ["2", "1", "3", "1", "2", "2"],
            "amount": [10.0, 5.0, 4.0, 15.0, 5.0, 100],
            "date": ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-04", "2019-01-05", "2019-01-01"],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": ["1", "1"],
            "balance": [100.0, 200.0],
            "date": ["2019-01-01", "2019-01-05"],
        })

        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans,
                                                     balance_updates=balance_updates, account_key="1")
        daily_bals = hist.get_daily_balance()
        print(daily_bals)
        mult = 0.18904171541952763 + 1.0
        exp = pd.DataFrame({
            "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05"]),
            "balance": [
                100.0,
                100.0 * mult,
                100.0 * mult * mult - 9.0,
                (100.0 * mult * mult - 9.0) * mult + 15.0,
                ((100.0 * mult * mult - 9.0) * mult + 15.0) * mult - 5.0
            ],
        }).set_index("date")

        np.testing.assert_equal(exp.index.values, daily_bals.index.values)
        np.testing.assert_array_almost_equal(exp["balance"].values, daily_bals.values)

    def test_get_daily_balances_edge_case_1(self):
        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": ["0", "0"],
            "to_account_key": ["1", "1"],
            "amount": [90.0, 90.0],
            "date": ["2019-01-03", "2018-12-30"],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": [],
            "balance": [],
            "date": [],
        })
        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans,
                                                     balance_updates=balance_updates, account_key="1")
        daily_bals = hist.get_daily_balance()
        exp = pd.DataFrame({
            "date": pd.to_datetime(["2018-12-29", "2018-12-30", "2018-12-31", "2019-01-01", "2019-01-02","2019-01-03"]),
            "balance": [0.0, 90.0, 90.0, 90.0, 90.0, 180.0],
        }).set_index("date")

        np.testing.assert_equal(exp.index.values, daily_bals.index.values)
        np.testing.assert_array_almost_equal(exp["balance"].values, daily_bals.values)

    def test_get_daily_balances_with_early_trans(self):
        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": ["0", "0"],
            "to_account_key": ["1", "1"],
            "amount": [90.0, 90.0],
            "date": ["2019-01-03", "2018-12-30"],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": ["1", "1"],
            "balance": [100.0, 200.0],
            "date": ["2019-01-01", "2019-01-05"],
        })

        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans,
                                                     balance_updates=balance_updates, account_key="1")
        daily_bals = hist.get_daily_balance()
        mult = 1.05409255338945984
        mult2 = 1.01689832725085294
        exp = pd.DataFrame({
            "date": pd.to_datetime(["2018-12-29", "2018-12-30", "2018-12-31", "2019-01-01", "2019-01-02",
                                    "2019-01-03", "2019-01-04", "2019-01-05"]),
            "balance": [
                0.0,  # 29
                90.0,  # 30
                90.0 * mult,  # 31
                90.0 * mult * mult,  # 01
                100.0 * mult2,  # 02
                100.0 * mult2 * mult2 + 90.0,  # 03
                (100.0 * mult2 * mult2 + 90.0) * mult2,  # 04
                (100.0 * mult2 * mult2 + 90.0) * mult2 * mult2,  # 05
            ],
        }).set_index("date")

        np.testing.assert_equal(exp.index.values, daily_bals.index.values)
        np.testing.assert_array_almost_equal(exp["balance"].values, daily_bals.values)

    def test_daily_interest_amounts(self):
        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": ["0", "0"],
            "to_account_key": ["1", "1"],
            "amount": [90.0, 90.0],
            "date": ["2019-01-03", "2018-12-30"],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": ["1", "1"],
            "balance": [100.0, 200.0],
            "date": ["2019-01-01", "2019-01-05"],
        })

        exp = pd.DataFrame({
            "date": pd.to_datetime(["2018-12-29", "2018-12-30", "2018-12-31", "2019-01-01", "2019-01-02",
                                    "2019-01-03", "2019-01-04", "2019-01-05"]),
            "interest": [0.0,0.0, 4.868330, 10.0-4.868330, 1.689833, 193.408221-90.0-101.689833, 196.676496-193.408221,
                         200.0-196.676496],
        }).set_index("date")["interest"]

        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans,
                                                     balance_updates=balance_updates, account_key="1")
        print(hist.get_daily_balance())
        print(hist.get_daily_transfers())

        daily_interest = hist.get_daily_interest()
        print(daily_interest)

        np.testing.assert_equal(exp.index.values, daily_interest.index.values)
        np.testing.assert_array_almost_equal(exp.values, daily_interest.values, 4)

    def test_daily_summary(self):

        bal_trans = BalanceTransfers.from_dict({
            "from_account_key": ["0", "0"],
            "to_account_key": ["1", "1"],
            "amount": [90.0, 90.0],
            "date": ["2019-01-03", "2018-12-30"],
        })

        balance_updates = BalanceUpdates.from_dict({
            "account_key": ["1", "1"],
            "balance": [100.0, 200.0],
            "date": ["2019-01-01", "2019-01-05"],
        })

        hist = MoneyFrame.from_updates_and_transfers(balance_transfers=bal_trans, balance_updates=balance_updates, account_key="1")
        print(hist.to_df())


if __name__ == '__main__':
    unittest.main()
