import logging
import unittest

import moneytrack as mt

logging.basicConfig(level=logging.DEBUG)


class TestCore(unittest.TestCase):

    def test_money_track(self):
        print(mt.MODULE_PATH)
        mt.MoneyData.from_csv_dir(mt.MODULE_PATH + "/../sample_data/")

    def test_money_track_excel(self):
        print(mt.MODULE_PATH)
        mt.MoneyData.from_excel(mt.MODULE_PATH + "/../sample_data/data.xlsx")

    def test_filter_groupby(self):
        d = {
            "A": mt.MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "B": mt.MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
            "C": mt.MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0),
        }
        mf_gb = mt.MoneyFrameCollection(d, mt.DataField.ACCOUNT_KEY)

        accounts = mt.Accounts.from_dict({
            mt.DataField.ACCOUNT_KEY: ["A", "B", "C"],
            mt.DataField.ISA: [True, True, False],
            mt.DataField.ACCOUNT_TYP: ["CURRENT", "S&S", "S&S"],
        })
        md = mt.MoneyData(accounts, mf_gb)

        mf_100 = mt.MoneyFrame.from_fixed_rate(days=5, start_bal=100.0, ayr_prcnt=5.0)
        mf_200 = mt.MoneyFrame.from_fixed_rate(days=5, start_bal=200.0, ayr_prcnt=5.0)

        self.assertEqual(md.groupby_accounts(mt.DataField.ISA)[True], mf_200)
        self.assertEqual(md.groupby_accounts(mt.DataField.ISA)[False], mf_100)
        self.assertEqual(md.filter_accounts({mt.DataField.ACCOUNT_TYP: "S&S"})
                         .groupby_accounts(mt.DataField.ISA)[True], mf_100)
        self.assertEqual(md.filter_accounts({mt.DataField.ACCOUNT_TYP: "S&S"})
                         .groupby_accounts(mt.DataField.ISA)[False], mf_100)

if __name__ == '__main__':
    unittest.main()
