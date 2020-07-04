import logging
import unittest

from moneytrack import MoneyData, MODULE_PATH

logging.basicConfig(level=logging.DEBUG)


class TestCore(unittest.TestCase):

    def test_money_track(self):
        print(MODULE_PATH)
        mt = MoneyData.from_csv_dir(MODULE_PATH + "/../sample_data/")

    def test_money_track_excel(self):
        print(MODULE_PATH)
        mt = MoneyData.from_excel(MODULE_PATH + "/../sample_data/data.xlsx")


if __name__ == '__main__':
    unittest.main()
