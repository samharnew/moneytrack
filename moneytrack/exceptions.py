
class BalanceExtrapolationError(Exception):
    """An error that is raised when the daily account balances cannot be
    extrapolated between two dates"""
    pass

class NoSolutionFoundError(Exception):
    pass