# streamlit framework
from random import sample
import streamlit as st

# data wrangling
import pandas as pd
import numpy as np

# utils
import pickle

from torch import embedding


@st.cache(allow_output_mutation=True)
def read_tweet_data(path='data/Tweets.csv'):
    tweets_df = pd.read_csv(path)[['tweet_id', 'airline', 'airline_sentiment', 'text']]
    return tweets_df


@st.cache()
def get_tweets_embeddings(path='data/bert_embeddings.npy'):
    return np.load(path)


@st.cache()
def get_svm_model(path='data/SVM_clf.pkl'):
    with open(path, 'rb') as f:
        clf = pickle.load(f)
    return clf


# loading data
tweets_df = read_tweet_data()
tweets_emb = get_tweets_embeddings()
svm_clf = get_svm_model()

label_encoder = {
    0:'negative',
    1:'neutral',
    2:'positive'
}

# app
st.title('Airline Tweet Sentiment Analysis')

generate_button = st.button('Sample a tweet')

if generate_button:
    sample_tweet = tweets_df.sample(1)
    sample_tweet_emb = tweets_emb[sample_tweet.index.tolist()[0]]
    predict_sentiment_tweet = svm_clf.predict(sample_tweet_emb.reshape(1, -1))
    predict_description_sentiment_tweet = label_encoder[predict_sentiment_tweet[0]]

    st.subheader('Tweet')
    st.write(sample_tweet['text'].iloc[0])

    st.subheader("Our prediction of the tweet's sentiment:")
    st.write(predict_description_sentiment_tweet)

    st.subheader("The labeled sentiment of the tweet:")
    st.write(sample_tweet['airline_sentiment'].iloc[0])
