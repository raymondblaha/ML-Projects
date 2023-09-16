import pandas as pd
from bs4 import BeautifulSoup
import sys
import requests as req
import dateutil.parser as parser
import os

# Load in the data
path = sys.argv[1]
df = pd.read_csv(path)
df.dropna(subset=['url'], inplace=True)

# Format the publishedAt column
df['date'] = df['date'].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d'))

# Create a function to extract text and publication date from the links
def extract_text_and_date(url):
    if not url.startswith('http'):
        url = 'https://' + url
    response = req.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text().replace('\n','').replace('\r','').strip()
    date_string = ''
    try:
        date_string = soup.find('time')['datetime']
    except:
        pass
    try:
        if not date_string:
            date_string = soup.find('span', {'class': 'timestamp'}).text.strip()
    except:
        pass
    try:
        if not date_string:
            date_string = soup.find('time', {'class': 'dateline'}).text.strip()
    except:
        pass
    if date_string:
        date = parser.parse(date_string)
        pub_date = date.strftime('%Y-%m-%d')
    else:
        pub_date = ''
    print(f'URL: {url}, Publication Date: {pub_date}')
    return text, pub_date

# Loop through the dataframe and fill in the new text column
df['text'] = ""
for index in df.index:
    url = df['url'][index]
    text, pub_date = extract_text_and_date(url)
    df.at[index, 'text'] = text
    print(f"Processed {index} out of {len(df)}")
    
# Save into a new csv file with new columns for text and publication_date
df.to_csv('new_' + path, index=False)
