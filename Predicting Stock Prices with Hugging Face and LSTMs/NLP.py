import pandas as pd
import numpy as np
import os 
import transformers
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt

# load in the data
sys.argv[1]
df = pd.read_csv(sys.argv[1])

# Load in the model
model = transformers.TFAutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load in the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Add columns for sentiment, probability, label, and score
df['sentiment'] = ""
df['probability'] = ""
df['label'] = ""
df['score'] = ""

# Create a function to encode the text
def encode_text(text):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors='tf'
    )
    return encoded_text


# Create a function to make predictions
def predict(text):
    encoded_text = encode_text(text)
    output = model(encoded_text)
    return output

# Create a function to get the sentiment
def get_sentiment(text):
    output = predict(text)
    sentiment = np.argmax(output[0])
    return sentiment

# Create a function to get the probability
def get_probability(text):
    output = predict(text)
    probability = tf.nn.softmax(output[0][0], axis=0)
    return probability

# Create a function to get the label being positive, negative, or neutral
def get_label(text):
    sentiment = get_sentiment(text)
    if sentiment <= 1:
        label = 'negative'
    elif sentiment >= 3:
        label = 'positive'
    else:
        label = 'neutral'
    return label


# Create a function to get the score
def get_score(text):
    probability = get_probability(text)
    score = probability[get_sentiment(text)]
    return score.numpy().item()

# Create a function to get the sentiment, probability, label, and score
def get_sentiment_analysis(text):
    sentiment = get_sentiment(text)
    probability = get_probability(text)
    label = get_label(text)
    score = get_score(text)
    return sentiment, probability, label, score

# Loop through the dataframe and fill in the new columns
for index in df.index:
    text = df['text'][index]
    sentiment, probability, label, score = get_sentiment_analysis(text)
    df.at[index, 'sentiment'] = sentiment
    df.at[index, 'probability'] = probability
    df.at[index, 'label'] = label
    df.at[index, 'score'] = score
    print(sentiment, probability, label, score)
    
# convert the probability from tensor to a list
df['probability'] = df['probability'].apply(lambda x: x.numpy().tolist())
    
# Save into a new csv file with new columns for sentiment, probability, label, and score
df.to_csv('new_' + sys.argv[1])





