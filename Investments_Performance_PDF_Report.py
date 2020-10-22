# In this program we take CSVs that are prepared with a search name - either a Fund name / ISIN / Stock ticker
# and use that to search either Investing.com (InvestPy) or Yahoo Finance (with pandas URL) to get historical
# price data. Then we plot graphs using matplotlib, and present these in PDF using ReportLab.
import investpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, LinearLocator
from urllib.error import HTTPError
from time import sleep
import os
import textwrap
import pickle  # Use pickle module to save complex Python objects to the disk. If all objects can be handled by json,
# you can use json module, which gives a human-readable file. However in this case, we have dataframes.
# You can read dfs into json, but they must be assigned as json first. Simpler to use pickle here.

# ReportLab imports
from reportlab.platypus import SimpleDocTemplate, PageTemplate, Frame, Flowable, Paragraph, Table, TableStyle, Spacer, KeepTogether  # Platypus contains the flowable classes
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # StyleSheet is a set of default style we can use, and ParagraphStyle can customise them
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER  # Import text alignment & justify constants here
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics  # Used to register fonts
from reportlab.pdfbase.ttfonts import TTFont  # Used for creating TrueType font object
from io import BytesIO  # IO streams are file-like objects that live in python memory, rather than on disk. Much faster, and less files!
from svglib.svglib import svg2rlg  # Library for converting SVG image files into other formats (e.g. ReportLab graphics.)

# Read csv file of trades for a tax year. CSVs must be prepared with a search name col that works on Investing.com or Yahoo.
equities_DF = pd.read_csv('./Investments CSV Example.csv', sep=',', header=0, index_col=0).dropna(how='all')

# Convert date strings to datetime objects.
equities_DF['Bought'] = pd.to_datetime(equities_DF['Bought'], format='%d/%m/%Y')
equities_DF['Sold'] = pd.to_datetime(equities_DF['Sold'], format='%d/%m/%Y')
# Calculate time buffer so we get a bit of extra price data before and after trade.
equities_DF['Buffer'] = (equities_DF['Sold'] - equities_DF['Bought']) * 0.2
# Create column for time interval between Bought and Sold
equities_DF['Interval'] = equities_DF['Sold'] - equities_DF['Bought']
# Create search-date columns for 'bought' and 'sold' with added buffer
equities_DF['Bought_search'] = equities_DF['Bought'] - equities_DF['Buffer']
equities_DF['Sold_search'] = equities_DF['Sold'] + equities_DF['Buffer']

# Create two Epoch timestamp (ts) columns for bought and sold dates.
equities_DF['Bought_ts'] = ((equities_DF['Bought_search'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')).astype('Int64')  # Int64 is a special pandas type that supports nullable Ints.
equities_DF['Sold_ts'] = ((equities_DF['Sold_search'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')).astype('Int64')

# Create master dictionary for holding name of equity and df of historical prices
prices = {}
# Function for fetching historical price data
def fetch_prices():
    consecutive_trades = ''  # Variable for checking consecutive trades (i.e. where two different purchases were sold together)
    consecutive_trades_I = ''  # Another variable for checking consecutive trades (but in second part of code)
    for equity in enumerate(equities_DF.index):
        # Add fund/share as dict key, and add sub-dict with key as 'Investpy fund name' / 'ISIN' / 'Ticker' and search name as value.
        if equity[1] not in prices:
            prices[equity[1]] = equities_DF.iloc[equity[0], 7:10].dropna().to_dict()
            consecutive_trades = equity[1]
        elif equity[1] == consecutive_trades:  # If a consecutive buy exists, add the date of that buy.
            prices[equity[1]]['Additional buy'] = equities_DF.iloc[equity[0], 1]
            consecutive_trades = equity[1]

        # Set default search country as UK, unless 'USD' found.
        country = 'United Kingdom'
        if 'USD' in equity[1]:
            country = 'United States'
        elif 'CAD' in equity[1]:
            country = 'Canada'

        # Retrieve historic fund/share prices
        # First check what type of search we need to do: using Fund Name, ISIN, or ticker
        if equity[1] == consecutive_trades_I:  # Skip the additional buys if they are part of same sell transaction.
            print(f'{equity[0]}. Additional buy for {equity[1]} - skipped.')
            continue
        elif 'InvestPy Fund Name' in prices[equity[1]]:
            search_name = prices[equity[1]]['InvestPy Fund Name']  # Get value that we use to search InvestPy or Yahoo.
            try:  # Add a df of historical price data to 'Price History' key
                prices[equity[1]]['Price History'] = investpy.get_fund_historical_data(fund=search_name,
                                                                                       country=country,  # Below converts datetime to string format for searching
                                                                                       from_date=equities_DF.iloc[equity[0], -4].strftime('%d/%m/%Y'),
                                                                                       to_date=equities_DF.iloc[equity[0], -3].strftime('%d/%m/%Y'),
                                                                                       interval='Daily')
                print(f'{equity[0]}. Retrieved fund price data for {equity[1]}.')
            except RuntimeError:
                print(RuntimeError, f'CHECK! InvestPy did not find price data for {equity[1]}')

        elif 'Stock Ticker' in prices[equity[1]]:
            search_name = prices[equity[1]]['Stock Ticker']
            try:
                prices[equity[1]]['Price History'] = investpy.get_stock_historical_data(stock=search_name,
                                                                                        country=country,
                                                                                        from_date=equities_DF.iloc[equity[0], -4].strftime('%d/%m/%Y'),
                                                                                        to_date=equities_DF.iloc[equity[0], -3].strftime('%d/%m/%Y'),
                                                                                        interval='Daily')
                print(f'{equity[0]}. Retrieved stock price data for {equity[1]}.')
            except RuntimeError:  # If InvestPy fails, try Yahoo Finance.
                prices[equity[1]]['Price History'] = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{search_name}?period1={equities_DF.iloc[equity[0], -2]}&period2={equities_DF.iloc[equity[0], -1]}&interval=1d&events=history', index_col='Date')
                # Yahoo Finance data not downloaded as datetime objects - convert these:
                prices[equity[1]]['Price History'].index = pd.to_datetime(prices[equity[1]]['Price History'].index, format='%Y-%m-%d')
                print(f'{equity[0]}. Retrieved stock price data for {equity[1]} from YF.')
                sleep(1)  # Ensures we don't overload Yahoo with requests.
            except HTTPError:
                print('CHECK! Yahoo Finance request failed for', equity[1])

        elif 'ISIN for Yahoo Finance' in prices[equity[1]]:
            search_name = prices[equity[1]]['ISIN for Yahoo Finance']
            try:
                prices[equity[1]]['Price History'] = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{search_name}?period1={equities_DF.iloc[equity[0], -2]}&period2={equities_DF.iloc[equity[0], -1]}&interval=1d&events=history', index_col='Date')
                prices[equity[1]]['Price History'].index = pd.to_datetime(prices[equity[1]]['Price History'].index, format='%Y-%m-%d')  # Convert index to datetime
                print(f'{equity[0]}. Retrieved fund price data for {equity[1]} using ISIN.')
                sleep(1)
            except HTTPError:
                try:  # Some ISIN numbers require a '.L' on the end to be found on Yahoo for some reason.
                    prices[equity[1]]['Price History'] = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{search_name}.L?period1={equities_DF.iloc[equity[0], -2]}&period2={equities_DF.iloc[equity[0], -1]}&interval=1d&events=history', index_col='Date')
                    prices[equity[1]]['Price History'].index = pd.to_datetime(prices[equity[1]]['Price History'].index, format='%Y-%m-%d')  # Convert index to datetime
                    print(f'{equity[0]}. Retrieved fund price data for {equity[1]} using ISIN.')
                    sleep(1)
                except HTTPError:
                    print('CHECK! Yahoo Finance request failed for', equity[1])
            except Exception as UnknownError:
                print('Unknown error for', equity[1], UnknownError)

        else:  # I couldn't find this equity on Investing.com or Yahoo Finance so we just skip it.
            print(f'{equity[0]}. No price data for this equity - skipped.')
        consecutive_trades_I = equity[1]  # Overwrite this var to check for consecutives.
        # Now correct price data which is in £ not pennies: Some funds randomly change from £s to pennies midway through dataset.
        try:  # Correct values which are < max value divided by 100.
            prices[equity[1]]['Price History'].loc[prices[equity[1]]['Price History']['Close'] < prices[equity[1]]['Price History']['Close'].max() / 100, ['Open', 'High', 'Low', 'Close']] *= 100
        except KeyError:
            print(KeyError, 'This equity had no price data')

# Fetch the prices if not found already:
if not os.path.isfile('./prices_2019-20.pkl'):
    fetch_prices()

# Save prices dictionary to disk, so I don't have to retrieve price data everytime.
# Highest protocol ensures the correct compatibility with my Python version. This is a binary encoding, hence 'wb'.
def save_prices(prices_dict, filename):
    with open(filename, 'wb') as filepath:
        pickle.dump(prices_dict, filepath, pickle.HIGHEST_PROTOCOL)
# Save the prices to file (can # out so it doesn't run everytime):
if not os.path.isfile('./prices_2019-20.pkl'):
    save_prices(prices, 'prices_2019-20.pkl')

# Read pickle file into Python again.
def load_prices(filename):
    with open(filename, 'rb') as file:
        prices = pickle.load(file)
    return prices
# Load the prices data
if os.path.isfile('./prices_2019-20.pkl'):
    load_prices('prices_2019-20.pkl')


###------------------------MATPLOTLIB PLOTTING SECTION------------------------###

# Create overview of trades in subplots. Create fig handle and axs 2D numpy array containing all 20 axes.
def overview_plots():
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(12, 6), tight_layout=True)
    fig.suptitle(f'Historical Price Data for Investments Sold in XXXX-XX')
    # Set accuracy of Tick labels to be used, depending on Buy-Sell time interval
    monthYear = mdates.DateFormatter('%b-%y')
    dayMonthYear = mdates.DateFormatter('%d/%m/%y')
    # ax.flat is an attribute of ax that gives an iterator where the 4x6 array is flattened to a 1D array. Allows us to loop through.
    for ax, (equity_name, equity) in zip(axs.flat, prices.items()):
        if equity.get('Price History') is not None:
            ax.set_title("\n".join(textwrap.wrap(equity_name, 45)), fontsize=6, wrap=True)  # Use textwrap to split string according to the char limit (45), then join with /n.
            ax.plot(equity['Price History'].index, equity['Price History']['Close'], color='blue', linewidth=1)
            ax.tick_params(labelsize=4)
            ax.set_xlabel('Date', fontsize=5)
            ax.set_ylabel('Price', fontsize=5)
            locator = MaxNLocator(nbins='auto')  # Create an automatic tick spacer
            numticks = LinearLocator(numticks=6)  # Create a linear tick spacer of set no. of ticks
            ax.yaxis.set_major_locator(locator)
            ax.xaxis.set_major_locator(numticks)
            # We use the 'Interval' column to determine what Tick formatting accuracy we should use on the graphs.
            interval = equities_DF.loc[equity_name, 'Interval']
            if isinstance(interval, pd.Series):  # Where we have consecutive trades, we have 2 values in a series.
                interval = equities_DF.loc[equity_name, 'Interval'][0]
            if interval < pd.Timedelta(60, 'days'):
                ax.xaxis.set_major_formatter(dayMonthYear)
                ax.tick_params(axis='x', labelrotation=30)
            else:
                ax.xaxis.set_major_formatter(monthYear)
            # Define buy and sold dates
            bought_date = equities_DF.loc[equity_name, 'Bought']
            sold_date = equities_DF.loc[equity_name, 'Sold']
            if isinstance(bought_date, pd.Series):
                bought_date = bought_date[0]
                sold_date = sold_date[0]
            # Try to annotate Buy and Sell arrows
            bought_ycoord = prices[equity_name]['Price History'].loc[bought_date, 'Close']
            sold_ycoord = prices[equity_name]['Price History'].loc[sold_date, 'Close']
            if not pd.isna([bought_ycoord, sold_ycoord]).any():
                ax.annotate('Bought', (bought_date, bought_ycoord), xycoords='data', fontsize=5, fontweight='semibold', color='orange', xytext=(-15, -25), textcoords='offset points', arrowprops={'arrowstyle': '->'})
                ax.annotate('Sold', (sold_date, sold_ycoord), xycoords='data', fontsize=5, fontweight='semibold', color='red', xytext=(-15, -25), textcoords='offset points', arrowprops={'arrowstyle': '->'})
            else:
                pass
        else:
            continue

overview_plots()


##########################################################################################################
###------------------------------------------ PDF Production ------------------------------------------###
##########################################################################################################
# Using ReportLab, you can either layout the PDF using a canvas, and painting it with static content, such as
# strings, lines, drawings, logos etc. Or you you can use Flowables which is a list of items or content that we want to add to the PDF.
# These are easily styled with margins, paragraph style etc., and so are great for report content that's used repeatedly.
# Flowables are appended one after the other, a bit like typing in a Word Doc. Whereas, static elements are drawn in a fixed location.
# Normally flowables are appended to the story list, which is then used to build the final PDF.
# Mixing static content, and flowables can be a bit messy though. The way to do it is to use PageTemplate, which draws on static
# content, and also has a Frame that holds the flowables. You assign that template to the PDF before building it.

# First, define function that draws static content. i.e. content that is in the same position for every page.
# This function is used later in drawOn argument, and MUST include (canvas, doc) args
def draw_static_elements(canvas, pdf_doc):
    canvas.saveState()  # saveState saves current font, graphics transform for later recall by the next restoreState
    # TrueType (.ttf) fonts are those used on Mac and PC systems, as opposed to Type1 fonts developed by Adobe in their PDFs.
    # Must use a font with .ttc, .ttf, .otf format. ReportLab searches through your computer for them. 'Font Suitcase' not usable unfortunately
    pdfmetrics.registerFont(TTFont('Euphemia', 'EuphemiaCAS.ttc'))
    canvas.setFont('Euphemia', 10)
    # Draw string at fixed location (top-left corner)
    canvas.drawString(30, 810, f'Report generated on {pd.to_datetime("today"):%d/%m/%Y}')
    # Reset font, graphic settings back to what they were before this function ran
    canvas.restoreState()

# Define function to rescale drawing objects
def scale_to_fit(drawing, pdf_doc):
    """This function scales the drawing object to fit within the margin width of the pdf SampleDocTemplate"""
    max_width = pdf_doc.width
    scale_factor = max_width / drawing.width
    # Not sure why, but width and height attributes don't actually change the size, but they must be changed to help the positioning during pdf build.
    drawing.width *= scale_factor
    drawing.height *= scale_factor
    drawing.scale(scale_factor, scale_factor)  # This actually scales the image by changing transform attr. Two args: scale_x, scale_y
    drawing.hAlign = 'RIGHT'
    return drawing

class Line(Flowable):  # Inherits attributes from Flowable class, so it can be appended to story.
    def __init__(self, width, height=0):  # Only need to specify width to draw a line.
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Line with width={self.width}"

    def draw(self):
        """Use canvas.line method. Provide two X,Y pairs for start and end of line."""
        self.canv.line(0, self.height, self.width, self.height)

line = Line(438)  # 438 is the approx width of the text in the PDF

# SET UP PDF READY FOR TAKING FIGURES #
# The simple doc template sets up our document. You can specify page size, margins etc
pdf = SimpleDocTemplate('Report Preview.pdf', topMargin=57, bottomMargin=35, author='Your Name', showBoundary=False)
# Create Frame for holding flowables. Frame object is used by the platypus modules. Args: x,y (bottom left),
frame = Frame(pdf.leftMargin, pdf.bottomMargin, pdf.width, pdf.height, showBoundary=False)
# Add Frame to the page template and call on template to draw static objects
template = PageTemplate(frames=[frame], onPage=draw_static_elements)
# Add the template to the simple doc
pdf.addPageTemplates(template)
# Get the preset paragraph/text styles
styles = getSampleStyleSheet()
# TrueType (.ttf) fonts are those used on Mac and PC systems, as opposed to Type1 fonts developed by Adobe in their PDFs.
# Must use a font with .ttc, .ttf, .otf format. ReportLab searches through your computer for them. 'Font Suitcase' not usable unfortunately
pdfmetrics.registerFont(TTFont('Palatino Linotype', 'Palatino Linotype.ttf'))
# Create custom paragraph style
styles.add(ParagraphStyle(name='MainTitle', fontName='Palatino Linotype', underlineWidth=1, fontSize=16, alignment=TA_CENTER))
styles.add(ParagraphStyle(name='EquityHeading', fontName='Palatino Linotype', fontSize=12, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='Body', fontName='Palatino Linotype', fontSize=10, alignment=TA_JUSTIFY))
# Define story list for holding flowables
story = list()
# Add a paragraph to the pdf story with the title. </u> is XML for underline.
story.append(Paragraph('<u>HL Fund and Share Account Trades: Tax Year XXXX-XX</u>', style=styles['MainTitle']))
# Add a blank line. If font size is 10, then height=12 adds a blank line.
story.append(Spacer(5, 30))

# In loop below, recreate individual, larger figures for each equity.
# Set accuracy of Tick labels to be used, depending on Buy-Sell interval
monthYear = mdates.DateFormatter('%b-%y')
dayMonthYear = mdates.DateFormatter('%d-%b-%y')
# Create historical price plots. Each plot will be saved in-memory to BytesIO object to be put into PDF document
for equity_name, equity in prices.items():
    if equity.get('Price History') is not None:
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        ax.plot(equity['Price History'].index, equity['Price History']['Close'], color='blue', linewidth=1)
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price', fontsize=11)
        locator = MaxNLocator(nbins='auto')
        numticks = LinearLocator(numticks=8)
        ax.yaxis.set_major_locator(locator)
        ax.xaxis.set_major_locator(numticks)
        # Use the Interval column to determine what Tick formatting accuracy we should use on the graphs.
        interval = equities_DF.loc[equity_name, 'Interval']
        if isinstance(interval, pd.Series):
            interval = equities_DF.loc[equity_name, 'Interval'][0]
        if interval < pd.Timedelta(60, 'days'):
            ax.xaxis.set_major_formatter(dayMonthYear)
        else:
            ax.xaxis.set_major_formatter(monthYear)
        # Try annotate Buy and Sell arrows
        bought_date = equities_DF.loc[equity_name, 'Bought']
        sold_date = equities_DF.loc[equity_name, 'Sold']
        if isinstance(bought_date, pd.Series):
            bought_date = bought_date[0]
            sold_date = sold_date[0]
        bought_ycoord = prices[equity_name]['Price History'].loc[bought_date, 'Close']
        sold_ycoord = prices[equity_name]['Price History'].loc[sold_date, 'Close']
        if not pd.isna([bought_ycoord, sold_ycoord]).any():
            try:
                ax.annotate('Bought', (bought_date, bought_ycoord), xycoords='data', fontsize=10, fontweight='semibold', color='orange', xytext=(-15, -70), textcoords='offset points', arrowprops={'arrowstyle': '->'})
                ax.annotate('Sold', (sold_date, sold_ycoord), xycoords='data', fontsize=10, fontweight='semibold', color='red', xytext=(-15, -70), textcoords='offset points', arrowprops={'arrowstyle': '->'})
            except KeyError:
                print(KeyError, equity_name)
        else:
            pass

        # ------------------------------- PDF construction ------------------------------- #
        # Create Bytes object (binary object) to save figure within Python. This avoids having to save file on disk
        chart = BytesIO()
        fig.savefig(chart, format='svg')
        # Set the current position of the file handle (like a cursor).
        # '0' sets cursor at beginning of file. So when we read file, we read from the start.
        chart.seek(0)
        # svg2rlg takes a SVG file and converts it to ReportLab graphic. Returns a drawing object. This can be directly appended to story.
        chartRL = svg2rlg(chart)
        chartRL = scale_to_fit(chartRL, pdf)

        # Define equity text to be appended later
        equityText = Paragraph(f'{equity_name}:', style=styles['EquityHeading'])
        # Define profit/loss number as float, and set green/pink colour for gain/loss. Also define return % number.
        profit_loss = equities_DF.loc[equity_name, 'Profit/Loss']
        return_pc = equities_DF.loc[equity_name, 'Return %']
        if isinstance(profit_loss, pd.Series):
            profit_loss = float(profit_loss[0].replace('£', '').replace(',', ''))
            return_pc = return_pc[0]
        else:
            profit_loss = float(profit_loss.replace('£', '').replace(',', ''))
        if profit_loss > 0:
            profit_loss_color = colors.palegreen
        else:
            profit_loss_color = colors.pink
        # Define table data. Each element of list is a row.
        table_data = [['', 'Bought', 'Sold', 'Profit/Loss', 'Return %'],
                      [equity_name, bought_date.strftime('%d/%m/%Y'), sold_date.strftime('%d/%m/%Y'), '£' + str(profit_loss), return_pc]]
        table = Table(table_data)
        # Set table style. (From cell) (To cell) (col, row)
        table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Palatino Linotype'),
                                   ('FONTSIZE', (0, 0), (-1, -1), 9),
                                   ('INNERGRID', (0, 0), (-1, -1), 1, colors.grey),
                                   ('BACKGROUND', (3, -1), (4, -1), profit_loss_color)]))

        # Use KeepTogether flowable to ensure line, spacer, chartRL etc. flowables all stay together for each equity.
        story.append(KeepTogether([line, Spacer(5, 6),
                                   equityText, Spacer(5, 4),
                                   chartRL, Spacer(5, 2),
                                   table, Spacer(5, 30)]))
    else:
        continue

# Close all plots
plt.close('all')

# Build pdf. Can also annotate page numbers, logos onto pages. Building pdf creates Canvas object, can be accessed by .canv
# Build pdf can also take onPage static drawing functions. Haven't tried this yet
pdf.build(story)