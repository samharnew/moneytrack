import logging
import os
import unittest

import pandas as pd

import moneytrack as mt

logging.basicConfig(level=logging.DEBUG)
field_names = mt.Config.FieldNames


class DatasetTests(unittest.TestCase):

    def test_a(self):
        sim = mt.simulation.AccountSimulatorFixedRate(date=pd.to_datetime("2021-01-01"), ayr=0.05, balance=100)
        for i in range(100):
            sim.step(0)
        print(sim.get_daily_account_history())

    def test_b(self):
        sim1 = mt.simulation.AccountSimulatorFixedRate(
            date=pd.to_datetime("2021-01-01"), ayr=0.05,
            balance=100, metadata={"ACCOUNT_KEY": "HSBC"}
        )
        sim2 = mt.simulation.AccountSimulatorFixedRate(
            date=pd.to_datetime("2021-01-01"), ayr=0.03,
            balance=100, metadata={"ACCOUNT_KEY": "BANKOFSAM"}
        )

        sim = mt.simulation.PortfolioSimulator([sim1, sim2])
        for i in range(5):
            sim.step()
        print(sim.get_daily_account_history())
