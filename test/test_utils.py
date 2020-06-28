import unittest
from moneytrack import *
import numpy as np
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_calc_real_pos_roots(self):
        roots = calc_real_pos_roots([1, 0, 0])
        np.testing.assert_array_equal(roots, [])

        roots = calc_real_pos_roots([1, 0, -1])
        np.testing.assert_array_equal(roots, [1])
        self.assertTrue(True)

    def test_create_daily_transfer_record(self):
        rec = create_daily_transfer_record([0, 3, 5], [10.3, 1.0, 3.2])
        np.testing.assert_array_equal(rec, [10.3, 0, 0, 1.0, 0, 3.2])

        rec = create_daily_transfer_record([0, 3, 3], [10.3, 1.0, 3.2])
        np.testing.assert_array_equal(rec, [10.3, 0, 0, 1.0 + 3.2])
        self.assertTrue(True)

    def test_calc_avg_interest_rate(self):
        rate = calc_avg_interest_rate(start_bal=1.0, end_bal=1.05, num_days=1, trans_days=[], trans_amts=[])
        self.assertAlmostEqual(rate, 0.05, 5)
        rate = calc_avg_interest_rate(start_bal=1.0, end_bal=1.05, num_days=2, trans_days=[], trans_amts=[])
        self.assertAlmostEqual(rate, np.sqrt(1.05) - 1.0, 5)
        rate = calc_avg_interest_rate(start_bal=0.5, end_bal=1.05, num_days=2, trans_days=[0], trans_amts=[0.5])
        self.assertAlmostEqual(rate, np.sqrt(1.05) - 1.0, 5)
        mult = calc_avg_interest_rate(start_bal=1.0, end_bal=2.05, num_days=3, trans_days=[1], trans_amts=[1.0]) + 1.0
        self.assertAlmostEqual(2.05, (mult * 1.0 + 1.0) * mult * mult, 7)
        mult = calc_avg_interest_rate(start_bal=1.0, end_bal=2.05, num_days=3, trans_days=[0], trans_amts=[1.0]) + 1.0
        self.assertAlmostEqual(2.05, 2.0 * mult * mult * mult, 7)

    def test_calc_avg_interest_rate_edge_case_1(self):
        mult = calc_avg_interest_rate(start_bal=0.0, end_bal=2.00, num_days=3, trans_days=[1], trans_amts=[2.0]) + 1
        self.assertAlmostEqual(2.0, mult*mult*2.0, 7)
        mult = calc_avg_interest_rate(start_bal=0.0, end_bal=2.00, num_days=3, trans_days=[0], trans_amts=[2.0]) + 1.0
        self.assertAlmostEqual(2.0, mult*mult*mult*2.0, 7)
        mult = calc_avg_interest_rate(start_bal=0.0, end_bal=2.05, num_days=3, trans_days=[1], trans_amts=[2.0]) + 1
        self.assertAlmostEqual(2.05, mult*mult*2.00, 7)
        mult = calc_avg_interest_rate(start_bal=0.0, end_bal=2.05, num_days=3, trans_days=np.array([0]),
                                      trans_amts=np.array([2.0])) + 1.0
        self.assertAlmostEqual(2.05, mult*mult*mult*2.00, 7)

    def test_calc_avg_interest_rate_edge_case_2(self):
        mult = calc_avg_interest_rate(start_bal=1.0, end_bal=2.00, num_days=3, trans_days=[3], trans_amts=[1.0]) + 1
        self.assertAlmostEqual(2.0, 1.0*mult*mult*mult + 1.0, 7)

    def test_calc_daily_balances(self):
        r = 1.01
        daily_bals = calc_daily_balances(100.0, 5, r - 1.0)
        np.testing.assert_array_almost_equal(daily_bals, [100.0, 100.0 * r, 100.0 * r * r, 100.0 * r * r * r,
                                                          100.0 * r * r * r * r], 7)
        daily_bals = calc_daily_balances(100.0, 5, r - 1.0, start_day=1)
        np.testing.assert_array_almost_equal(daily_bals, [0.0, 100.0, 100.0 * r, 100.0 * r * r, 100.0 * r * r * r], 7)

        self.assertTrue(True)

    def test_calc_daily_balances_w_transfers(self):
        r = 1.01
        daily_bals = calc_daily_balances_w_transfers(start_bal=1.0, daily_rate=r - 1.0, num_days=4)
        exp = [1.0, r, r * r, r * r * r]
        np.testing.assert_array_almost_equal(daily_bals, exp, 7)

        daily_bals = calc_daily_balances_w_transfers(start_bal=1.0, daily_rate=r - 1.0, num_days=4, trans_amts=[1.0],
                                                     trans_days=[1])
        exp = [1.0, 1.0 + r, (1.0 + r) * r, (1.0 + r) * r * r]
        np.testing.assert_array_almost_equal(daily_bals, exp, 7)

        # Check it works with numpy arrays too
        daily_bals = calc_daily_balances_w_transfers(start_bal=1.0, daily_rate=r - 1.0, num_days=4,
                                                     trans_amts=np.array([1.0]), trans_days=np.array([1]))
        exp = [1.0, 1.0 + r, (1.0 + r) * r, (1.0 + r) * r * r]
        np.testing.assert_array_almost_equal(daily_bals, exp, 7)
        self.assertTrue(True)

    def test_calc_daily_balances_w_transfers_no_interest(self):

        r=1.0
        daily_bals = calc_daily_balances_w_transfers(start_bal=0.0, daily_rate=r - 1.0, num_days=4, trans_amts=[1.0],
                                                     trans_days=[0])
        np.testing.assert_array_almost_equal(daily_bals, [1.0]*4, 7)


    def test_dates_between(self):
        start_date = pd.to_datetime("2020-01-01")
        end_date = pd.to_datetime("2020-01-03")

        dates = dates_between(start_date, end_date)
        self.assertListEqual(dates, list(map(pd.to_datetime, ["2020-01-01", "2020-01-02", "2020-01-03"])))

    def test_cross_join(self):
        df_a = pd.DataFrame({'a': [0, 1]})
        df_b = pd.DataFrame({'b': [0, 1]})
        df_exp = pd.DataFrame({'a': [0, 0, 1, 1], 'b': [0, 1, 0, 1]})
        pd.testing.assert_frame_equal(df_exp, cross_join(df_a, df_b))
        self.assertTrue(len(df_a.columns) == 1)



if __name__ == '__main__':
    unittest.main()
