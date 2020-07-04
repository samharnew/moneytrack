import logging
import unittest

from moneytrack import MoneyFrame, MoneyFrameCollection

logging.basicConfig(level=logging.DEBUG)


class TestMFCol(unittest.TestCase):

    def test_filter_groupby(self):
        d = {
            "a": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "b": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "c": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
        }

        mf_100 = MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0)
        mf_200 = MoneyFrame.from_fixed_rate(days=5, start_bal=200.0, ayr_prcnt=5.0)
        mf_300 = MoneyFrame.from_fixed_rate(days=5, start_bal=300.0, ayr_prcnt=5.0)

        mf_gb = MoneyFrameCollection(d, "account_key")

        self.assertTrue(mf_gb.sum() == mf_300)
        self.assertTrue(mf_gb.filter(lambda x: x in ["a", "b"]).sum() == mf_200)

        mf_gb2 = mf_gb.groupby([('a', 1), ('b', 1), ('c', 2)])
        self.assertTrue(mf_gb2.filter(lambda x: x == 1).sum() == mf_200)
        self.assertTrue(mf_gb2.filter(lambda x: x == 2).sum() == mf_100)

        d = dict([('a', 1), ('b', 1), ('c', 2)])
        mf_gb2 = mf_gb.groupby(d)
        self.assertTrue(mf_gb2.filter(lambda x: x == 1).sum() == mf_200)
        self.assertTrue(mf_gb2.filter(lambda x: x == 2).sum() == mf_100)

        mf_gb2 = mf_gb.groupby((['a', 'b', 'c'], [1, 1, 2]))
        self.assertTrue(mf_gb2.filter(lambda x: x == 1).sum() == mf_200)
        self.assertTrue(mf_gb2.filter(lambda x: x == 2).sum() == mf_100)

    def test_interest(self):
        d = {
            "a": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "b": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "c": MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
        }

        mf_gb = MoneyFrameCollection(d, "account_key")

        expected = {"a": 5.0, "b": 5.0, "c": 5.0}

        d = mf_gb.avg_interest_rates()
        for k, v in d.items():
            self.assertAlmostEqual(v, expected[k])

        d = mf_gb.groupby(lambda x: 1).avg_interest_rates()
        self.assertAlmostEqual(d[1], 5.0)

        df = mf_gb.groupby(lambda x: 1).avg_interest_rates(as_df=True)
        print(df)


if __name__ == '__main__':
    unittest.main()
