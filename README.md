## Investments Performance Report Generator
This is a Python script used for looking up the historical price action of your former investment holdings during the time that you owned them.
The program then generates a PDF report with plots of the price action during the time period (plus a little extra) in which you held those investments.

### Motivation
There are no brokers that I know of, Hargreaves Lansdown, Trading 212, etc. that provide an overview of price action of your previous investments.
It is always interesting to see the full picture of how an investment has performed while you held it. 

### Getting started
Since this program is not aimed at any broker in particular, it relies on you providing it with a `Investments CSV Example.csv` file in the correct format.
This `.csv` file will contain your investment history, with the following details:
![Image1](Images/CSV_File_Example.png?raw=true)
The Python program attempts to retrieve historical price data from one of two separate sources:
1. [Investing.com](https://www.investing.com/) (using Investpy)
2. [Yahoo Finance](https://uk.finance.yahoo.com/) (using Pandas)

Therefore, each investment requires ONE of the last 3 columns to be completed with a 'search' term.
- For stocks and shares, simply fill the 'Stock Ticker column'.
- For funds, either the 'ISIN for Yahoo Finance' or the 'InvestPy Fund Name' column can be filled out.
Not all funds can be found on Yahoo Finance using their ISIN, but if the fund can be found on Investing.com, price data can be retrieved through Investpy using the fund name exactly as it appears on Investing.com. 

#### Prerequisites
The dependencies for this script are:
- Python 3.6>
- Investpy
- Pandas, Matplotlib
- ReportLab
- svglib
- A correctly formatted `Investments CSV Example.csv` file.
#### To-Do
- Incorporate Financial Times price data search.
#### Licensing
This software is under the MIT license.
#### Contact
If you have any issues or ideas, don't hesitate to get in touch.