import pandas as pd
import numpy as np
import os
import sys

# load in the csv files: 
news = sys.argv[1]
prices = sys.argv[2]

# read in the csv files:
news_df = pd.read_csv(news, parse_dates=['date'])
prices_df = pd.read_csv(prices, parse_dates=['Date'])

# drop the following columns: symbol, title, description, url
news_df.drop(columns=['symbol', 'title', 'description', 'url'], inplace=True)


# Join the two dataframes together
df = pd.merge(prices_df, news_df, how='left', left_on='Date', right_on='date')

# drop the unnamed: 0 column
df.drop(columns=['Unnamed: 0'], inplace=True)


# print the new dataframe
print(df)

# Save the new dataframe into a new csv file
df.to_csv('FINAL_' + news, index=False)

