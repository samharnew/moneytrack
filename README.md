# moneytrack

Moneytrack is a python package that helps you track and monitor 
your personal finances.
As a user you provide updates of account balances over time
(as frequently or infrequently as you wish). 
You must also record any balance transfers between accounts. 
This information is processed to infer daily interest payments
per account, which can then be aggregated over different time periods
and/or account types. For example, you may wish to see your average
rate of interest over your entire portfolio for a given year.

## Installation

Moneytrack is available from PyPi, and can be installed
as follows:

```
pip install moneytrack
```

## Datasets

Moneytrack relies on three datasets that must be updated by the user.
These can be stored in separate csv files, or alternatively as
an Excel workbook with multiple sheets. 
Example data can be found in `moneytrack/sample_data`.
A brief description of each dataset follows:

#### Balance Updates 

This dataset is used to checkpoint account balances - the difference 
between checkpoints can be used to determine interest payments.

#### Balance Transfers

This dataset is used to record transfers between different accounts. 
If a transfer and and balance update are made on the same day, it is
assumed that the update is *after* the transfer. 

#### Accounts

This dataset records all accounts that are tracked 
in MoneyTrack. 
You must give each account a unique key that is used
in the Balance Updates / Balance Transfers data.
Here you can also give your accounts any number of properties 
e.g. you could add a boolean IS_ISA column,
which would allow you to determine your overall
interest payments from ISA/non-ISA accounts. 

## Tutorial

Once you have installed moneytrack and populated
your datasets (as described above). You are ready
to get started.

The best way to use moneytrack is from a Jupyter notebook.

### Loading data
If your data is stored in an Excel workbook, it can be loaded 
as follows:

```python
import moneytrack as mt
md = mt.MoneyData.from_excel("moneytrack_data.xlsx")
```

If you are instead storing the data in csv files (three 
separate files for balance updates, balance transfers and accounts),
they can be loaded as follows:

```python
import moneytrack as mt
md = mt.MoneyData.from_csv_dir("moneytrack_data/")
```

If the csv files are in separate directories, the `mt.MoneyData.from_csv` 
method allows you to specify the path to each dataset. 

### Filtering and aggregating

Once your `MoneyData` object has been loaded, you can apply filters
and / or aggregate your data. To filter for a specific date range:

```python
md_filtered = md["2021-01-02":"2021-04-01"]
```

There are also some helper functions for common date ranges e.g.

```python
import moneytrack as mt

date_range_2020 = mt.DateRanges.calendar_year(year=2020)
dates_from_2020_02 = mt.DateRanges.from_date(year=2020, month=2)
date_range_tax_20_21 = mt.DateRanges.tax_year(year_starting=2020)

md = mt.MoneyData.from_excel("moneytrack_data.xlsx")
md_filtered = md[date_range_tax_20_21]
```

If you wish to filter by any account property (any column in the
accounts data), then this is also possible. For example:

```python
md_stocks = md.filter_accounts({"ACCOUNT_TYP":"STOCKS AND SHARES"})
md_filtered = md.filter_accounts({"ACCOUNT_TYP":["STOCKS AND SHARES", "EASY_ACCESS"]})
```

If you wish to aggregate data, you can use the `groupby_accounts` method,
passing any account property. For example, the following will aggregate 
ISA and non-ISA accounts together:

```python
md_isa = md.groupby_accounts("ISA")
```

### Visualization 

To visualize your data there are two options `MoneyPlot` and `MoneyPlotly`.
The former uses matplotlib, whereas the latter uses Plotly 
(interactive plotting). The interface for both options is identical.

The following will show the amount of interest gained cumulatively over
time:

```python
f, ax = mt.MoneyPlot.plot(
    md,
    metric=mt.Metric.Interest,
    cumulative=True
)
ax.legend()
f.autofmt_xdate()
```

In addition to the 'Interest' metric, you can also display 'Balance', 'Transfers',
and 'InterestRate'. 

For some metrics, it is also possible to draw bar charts and pie charts:

```python
f, ax = mt.MoneyPlot.bar(md, mt.Metric.InterestRate, normalize=False)
```

```python
f, ax = mt.MoneyPlot.pie(md, mt.Metric.Balance, normalize=False)
```