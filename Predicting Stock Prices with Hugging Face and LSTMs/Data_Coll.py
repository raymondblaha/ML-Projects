import yfinance as yf
import pandas as pd
import datetime as dt
import requests as req
import dateutil.parser as parser
import numpy as np
import sys
import os


# Define the symbols list
symbols =['AAPL', 'MORF', 'LIND', 'TIO', 'RBLX', 'RAVE', 'AAU', 'NAK', 'HUGE', 'OGI', 'IGC', 'TRX', 'ACB',
          'FURY']

# Define the start and end dates
start_date = dt.datetime(2023, 3, 21)
end_date = dt.datetime(2023, 4, 17)

# Define the maximum number of articles per day
max_articles_per_day = 3

# API key
api_key = '00b667e3bb1640bfa44fe23fd79643a7'

# Define the news API endpoint
endpoint = 'https://newsapi.org/v2/everything'

# Loop through each stock symbol
for symbol in symbols:
    # Download the stock prices for the given date range
    prices = yf.download(symbol, start=start_date, end=end_date)

    # Save the stock prices to a CSV file with the stock symbol as the file name
    prices.to_csv(f"{symbol}.csv")

    # Initialize an empty dataframe to store the articles
    articles_df = pd.DataFrame(columns=["symbol", "date", "title", "description", "url"])

    # Loop through each day in the date range
    for day in pd.date_range(start=start_date, end=end_date):
        # Define the query parameters for the News API
        params = {
            'q': symbol,
            'apiKey': api_key,
            'from': day.isoformat(),
            'to': day.isoformat(),
            'pageSize': max_articles_per_day,
            'sortBy': 'publishedAt'
        }

        # Send a request to the News API
        response = req.get(endpoint, params=params)

        # Check if the response was successful
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(response.content)
        else:
            # Extract the articles from the response
            try:
                articles = response.json()["articles"]
            except KeyError:
                print("Error: no articles found in response")
                print(response.content)
            else:
                # Loop through each article
                for article in articles:
                    # Parse the publishedAt date
                    published_at = parser.parse(article["publishedAt"])

                    # Check if the publishedAt date is within the current day
                    if published_at.date() == day.date():
                        # Add the article to the dataframe
                        articles_df = articles_df._append({
                            "symbol": symbol,
                            "date": published_at,
                            "title": article["title"],
                            "description": article["description"],
                            "url": article["url"]
                        }, ignore_index=True)

    # Save the articles to a CSV file with the symbol as the filename
    articles_df.to_csv(f"{symbol}_articles.csv", index=False)
