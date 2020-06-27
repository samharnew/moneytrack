import logging
import unittest
import pandas as pd
import os
from moneytrack import Accounts, BalanceTransfers, BalanceUpdates, compare_pd_df, DataFields, SAMPLE_PATH

logging.basicConfig(level=logging.DEBUG)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.accounts_csv_path = os.path.join(SAMPLE_PATH, "accounts.csv")
        self.updates_csv_path = os.path.join(SAMPLE_PATH, "balance_updates.csv")
        self.transfers_csv_path = os.path.join(SAMPLE_PATH, "transfers.csv")
        self.excel_path = os.path.join(SAMPLE_PATH, "data.xlsx")

    def test_save_load_accounts_csv(self):
        accounts = Accounts.from_csv(self.accounts_csv_path)
        accounts.to_csv("tmp.csv")
        accounts_from_save = Accounts.from_csv("tmp.csv")
        pd.testing.assert_frame_equal(accounts.df, accounts_from_save.df)
        os.remove("tmp.csv")
        self.assertTrue(True)

    def test_save_load_updates_csv(self):
        updates = BalanceUpdates.from_csv(self.updates_csv_path)
        updates.to_csv("tmp.csv")
        updates_from_save = BalanceUpdates.from_csv("tmp.csv")
        pd.testing.assert_frame_equal(updates.df, updates_from_save.df)
        os.remove("tmp.csv")
        self.assertTrue(True)

    def test_save_load_transfers_csv(self):
        transfers = BalanceTransfers.from_csv(self.transfers_csv_path)
        transfers.to_csv("tmp.csv")
        transfers_from_save = BalanceTransfers.from_csv("tmp.csv")
        pd.testing.assert_frame_equal(transfers.df, transfers_from_save.df)
        os.remove("tmp.csv")
        self.assertTrue(True)

    def test_load_incorrect_csv(self):
        self.assertRaises((AssertionError), BalanceUpdates.from_csv, self.transfers_csv_path)
        self.assertRaises((AssertionError), BalanceUpdates.from_csv, self.accounts_csv_path)
        self.assertRaises((AssertionError), BalanceTransfers.from_csv,  self.accounts_csv_path)
        self.assertRaises((AssertionError), BalanceTransfers.from_csv,  self.updates_csv_path)
        self.assertRaises((AssertionError), Accounts.from_csv, self.updates_csv_path)
        self.assertRaises((AssertionError), Accounts.from_csv, self.transfers_csv_path)
        self.assertRaises((IOError), Accounts.from_csv, "missing_file.csv")


    def test_save_load_accounts_excel(self):
        accounts_1 = Accounts.from_excel(self.excel_path, "accounts")
        accounts_2 = Accounts.from_csv(self.accounts_csv_path)
        print(accounts_1)
        print(accounts_2)
        pd.testing.assert_frame_equal(accounts_1.df, accounts_2.df)

    def test_save_load_updates_excel(self):
        accounts_1 = BalanceUpdates.from_excel(self.excel_path, "balance_updates")
        accounts_2 = BalanceUpdates.from_csv(self.updates_csv_path)
        pd.testing.assert_frame_equal(accounts_1.df, accounts_2.df)

    def test_save_load_transfers_excel(self):
        accounts_1 = BalanceTransfers.from_excel(self.excel_path, "transfers")
        accounts_2 = BalanceTransfers.from_csv(self.transfers_csv_path)
        pd.testing.assert_frame_equal(accounts_1.df, accounts_2.df)

    def test_equals(self):
        balance_updates1 = BalanceUpdates.from_dict({
            DataFields.ACCOUNT_KEY: ["1", "2", "1"],
            DataFields.BALANCE: [100.0, 200.0, 200.0],
            DataFields.DATE: ["2019-01-01", "2019-01-01", "2019-01-03"],
        })
        balance_updates2 = BalanceUpdates.from_dict({
            DataFields.ACCOUNT_KEY: ["1", "2", "1"],
            DataFields.BALANCE: [100.0, 200.0, 200.0],
            DataFields.DATE: ["2019-01-01", "2019-01-01", "2019-01-03"],
        })
        balance_updates3 = BalanceUpdates.from_dict({
            DataFields.ACCOUNT_KEY: ["1", "1", "2"],
            DataFields.BALANCE: [100.0, 200.0, 200.0],
            DataFields.DATE: ["2019-01-01", "2019-01-03", "2019-01-01"],
        })
        balance_updates4 = BalanceUpdates.from_dict({
            DataFields.ACCOUNT_KEY: ["1", "2", "2"],
            DataFields.BALANCE: [100.0, 200.0, 200.0],
            DataFields.DATE: ["2019-01-01", "2019-01-03", "2019-01-01"],
        })
        self.assertTrue(balance_updates1.equals(balance_updates2))
        self.assertTrue(balance_updates1.equals(balance_updates3))
        self.assertFalse(balance_updates1.equals(balance_updates4))

    def test_get_acc_updates(self):
        balance_updates = BalanceUpdates.from_dict({
            DataFields.ACCOUNT_KEY: ["1", "2", "1"],
            DataFields.BALANCE: [100.0, 200.0, 200.0],
            DataFields.DATE: ["2019-01-01", "2019-01-01", "2019-01-03"],
        })
        df = balance_updates.get_acc_updates(account_key="1")
        exp = pd.DataFrame(
            {
                DataFields.BALANCE: [100.0, 200.0],
                DataFields.DATE: pd.to_datetime(["2019-01-01", "2019-01-03"]),
            }
        )
        self.assertTrue(compare_pd_df(df, exp, sort=False))

        df = balance_updates.get_acc_updates(account_key="1")
        exp = pd.DataFrame(
            {
                DataFields.BALANCE: [200.0, 100.0],
                DataFields.DATE: pd.to_datetime(["2019-01-03", "2019-01-01"]),
            }
        )
        self.assertTrue(compare_pd_df(df, exp, sort=True))

        df = balance_updates.get_acc_updates(account_key="1", prev_update_cols=True)
        exp = pd.DataFrame(
            {
                DataFields.BALANCE: [200.0, 100.0],
                DataFields.DATE: pd.to_datetime(["2019-01-03", "2019-01-01"]),
                DataFields.PREV_BALANCE: [100.0, None],
                DataFields.PREV_DATE: pd.to_datetime(["2019-01-01", None]),
            }
        )
        self.assertTrue(compare_pd_df(df, exp, sort=True))

    def test_get_acc_transfers(self):

        bal_trans = BalanceTransfers.from_dict({
            DataFields.FROM_ACCOUNT_KEY: ["1", "2", "2"],
            DataFields.TO_ACCOUNT_KEY: ["2", "1", "3"],
            DataFields.AMOUNT: [100.0, 200.0, 200.0],
            DataFields.DATE: pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]),
        })

        df = bal_trans.get_acc_transfers("1")
        print(df)

        exp = pd.DataFrame(
            {
                DataFields.AMOUNT: [-100.0, 200.0],
                DataFields.DATE: pd.to_datetime(["2019-01-01", "2019-01-02"]),
            }
        )
        self.assertTrue(compare_pd_df(df, exp, sort=True))

if __name__ == '__main__':
    unittest.main()
