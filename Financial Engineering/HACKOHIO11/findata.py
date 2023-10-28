import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# Define the stock symbol and date range
symbol = "AAPL"  # Replace with your desired stock symbol
start_date = datetime(2023, 1, 15)
end_date = start_date + relativedelta(months=6)

# Fetch historical data
df = yf.download(symbol, start=start_date, end=end_date, period="1d")
df['Date'] = df.index.strftime('%Y-%m-%d')

# Extract Open, Close, High, and Low prices into a Pandas DataFrame
price_data = df[['Date', 'Open', 'Adj Close', 'High', 'Low']]

price_df = pd.DataFrame(price_data)

# Print price_data
# print(price_data)
# print(price_data.size())

#price_data.to_excel('price_data1.xlsx', index=True)
json_data = json.dumps(price_data.to_dict(orient='records'))

# print(json_data)
