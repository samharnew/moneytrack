# MoneyTrack
## Datasets

#### BalanceUpdates

This data set is used to checkpoint account balances - the difference 
between checkpoints can be used to determine the interest.

#### BalanceTransfers

This data set is used to record transfers between different accounts. 
If a transfer and and balance update are made on the same day, it is
assumed that the update is *after* the transfer. 

#### Accounts

This data set is used to keep a record of all accounts that are tracked 
in MoneyTrack 
