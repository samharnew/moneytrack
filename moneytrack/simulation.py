from typing import Dict, List
import pandas as pd
from .utils import ayr_to_adr


class AccountSimulatorStep:

    # Would prefer to do this with a dataclass, but want to make it backwards
    # compatible with python 3.6
    balance: float
    date: pd.Timestamp
    transfers: float
    interest: float

    def __init__(self, balance: float, date: pd.Timestamp, transfers: float = 0.0, interest: float = 0.0):
        self.balance = balance
        self.date = date
        self.transfers = transfers
        self.interest = interest


class AccountSimulator:

    def __init__(self, date: pd.Timestamp, balance: float = 0, metadata: Dict[str, str] = None):
        self.balance = balance
        self.metadata = metadata if metadata is not None else dict()
        self.simulation_steps = [
            AccountSimulatorStep(balance=balance, date=date)
        ]
        self.date = date

    def _step(self, transfer) -> AccountSimulatorStep:
        pass

    def _increment_date(self):
        self.date += pd.to_timedelta(1, "d")

    def step(self, transfer) -> AccountSimulatorStep:
        self._increment_date()
        sim_step = self._step(transfer)
        self.simulation_steps.append(sim_step)
        return sim_step

    def get_daily_account_history(self) -> pd.DataFrame:
        df = pd.DataFrame([
            {
                "balance": step.balance,
                "date": step.date,
                "transfers": step.transfers,
                **self.metadata
            }
            for step in self.simulation_steps
        ])
        return df

class AccountSimulatorFixedRate(AccountSimulator):

    def __init__(self, date: pd.Timestamp, ayr: float, balance: float = 0, metadata: Dict[str, str] = None):
        super().__init__(date, balance, metadata)
        self.ayr = ayr

    def _step(self, transfer) -> AccountSimulatorStep:
        interest = self.balance*ayr_to_adr(self.ayr)
        self.balance += (interest + transfer)

        return AccountSimulatorStep(
            balance=self.balance,
            transfers=transfer,
            interest=interest,
            date=self.date
        )


class PortfolioSimulator:

    def __init__(self, account_simulators: List[AccountSimulator]):
        self.account_simulators = account_simulators

    def step(self):
        for acc in self.account_simulators:
            acc.step(0.0)

    def get_daily_account_history(self):

        return pd.concat([
            acc.get_daily_account_history()
            for acc in self.account_simulators
        ], axis=0)


    # def get_balance_updates(self, sample=1.0) -> pd.DataFrame:
    #
    #
    # def get_transfers(self):
    #
    #
    # def get_accounts(self):

