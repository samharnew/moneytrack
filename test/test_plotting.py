import os
import unittest

import moneytrack as mt


class TestPlotting(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_plt_dir = os.path.join(mt.TEST_PATH, "sample_plots")
        if not os.path.exists(self.sample_plt_dir):
            os.makedirs(self.sample_plt_dir)

    def test_mf_plot(self):
        mf = mt.MoneyFrame.from_fixed_rate(days=500, start_bal=100.0, ayr_prcnt=5.0)

        f, ax = mt.MoneyPlot.plot(mf, mt.Metric.Interest, cumulative=True)
        f.autofmt_xdate()
        f.savefig(os.path.join(self.sample_plt_dir, "mf_cum_interest.pdf"))

    def test_mfc_plot(self):
        d = {
            "a": mt.MoneyFrame.from_fixed_rate(days=40, start_bal=110.0, ayr_prcnt=5.0),
            "b": mt.MoneyFrame.from_fixed_rate(days=70, start_bal=100.0, ayr_prcnt=2.0),
            "c": mt.MoneyFrame.from_fixed_rate(days=50, start_bal=114.0, ayr_prcnt=-3.5),
        }

        mfc = mt.MoneyFrameCollection(d, "account_key")
        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.Balance, cumulative=False)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_balance.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.Transfers, cumulative=False)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_transfers.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.Transfers, cumulative=True)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_cum_transfers.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.InterestRate, cumulative=False)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_interest_rate.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.InterestRate, cumulative=True)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_cum_interest_rate.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.Interest, cumulative=False)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_interest.pdf"))

        f, ax = mt.MoneyPlot.plot(mfc, mt.Metric.Interest, cumulative=True)
        f.savefig(os.path.join(self.sample_plt_dir, "mfc_cum_interest.pdf"))


if __name__ == '__main__':
    unittest.main()
