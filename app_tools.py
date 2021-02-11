import pandas as pd
import pickle
import os
import glob
import re
import string
import emoji
import numpy as np
import matplotlib.pyplot as plt

import twint
import nest_asyncio
nest_asyncio.apply()

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool

from bs4 import BeautifulSoup

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def scrape_user(username):
    if username[0] == '@':
        username = username[1:]

    c = twint.Config()
    c.Hide_output = True
    c.Username = username
    c.Pandas = True
    c.Limit = 500
    c.Filter_retweets = False

    twint.run.Search(c)

    tweets_df = twint.storage.panda.Tweets_df

    columns_wanted = ['date', 'tweet']
    tweets_df_filtered = tweets_df[columns_wanted]

    return tweets_df_filtered

def clean_tweet_sentiment_analysis(tweet):

    #converts html to text
    tweet = BeautifulSoup(tweet, 'lxml').text

    #removes links
    tweet = re.sub(r'http\S+', '', tweet)

    #removes twitter usernames
    tweet = re.sub(r'(\s)?@\w+', '', tweet)

    return(tweet)

def calc_tweet_sentiment(tweet, sent_analyzer):
    return sent_analyzer.polarity_scores(tweet)['compound']

def clean_tweet_tfidf(tweet):

    custom_punctuation = '!"#&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
    custom_stop_words = [word.replace("'", "") for word in stopwords.words('english')] \
    + ['rt', 'amp', 'u', 'w', 'im', 'live', 'must', 'join', 'tune', 'pm', 'et',
    'year', 'say', 'get', 'it']

    tweet = tweet.lower()

    #removing punctuation
    cleaned_tweet = re.sub('[%s]' % re.escape(custom_punctuation), '', tweet)

    #removing emojis
    cleaned_tweet = re.sub(emoji.get_emoji_regexp(), r"", cleaned_tweet)

    #remove stop words
    cleaned_tweet = ' '.join([item for item in cleaned_tweet.split() if item not in custom_stop_words])

    return(cleaned_tweet)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_tweets(tweets):
    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    #lemmatize the words
    return [' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(tweet)]) for tweet in tweets]

def calc_word_vectors(df, words):
    word_scores = []
    for word in words:
        word_df = df[df['lemmatized_tweets'].str.contains(word)]
        if word_df.empty:
            word_scores.append(0)
        else:
            word_scores.append(word_df['sentiment'].mean())
    return(word_scores)

def create_similarity_matrix(similarity_tool, df_vectors):

    similarities = np.zeros((len(df_vectors), len(df_vectors)))

    for i in range(len(df_vectors)):
        politician_1 = df_vectors.iloc[i,:].values.reshape(1, -1)
        for j in range(i, len(df_vectors)):
            politician_2 = df_vectors.iloc[j, :].values.reshape(1, -1)
            similarities[i][j] = similarities[j][i] = similarity_tool(politician_1,politician_2)

    df = pd.DataFrame(similarities, index = df_vectors.index, columns = df_vectors.index)

    return df

def make_pca_plot(pcs, labels):
    source = ColumnDataSource({
        'x': pcs[:, 0],
        'y': pcs[:, 1],
        'label': labels
    })

    tools = ['tap', 'reset', HoverTool(tooltips = '@label')]

    p = figure(plot_width=700, plot_height=400, tools=tools)

    p.circle(x = 'x', y = 'y', size=10, source = source)
    #p.circle(x = 'x', y = 'y', legend_field = 'label', size=10, source = source)
    #p.add_layout(p.legend[0], 'right')

    p.xaxis.axis_label = 'PC1'
    p.yaxis.axis_label = 'PC2'
    p.legend.click_policy="hide"

    return(p)

def make_sim_bar_chart(matrix):

    tools = ['pan', 'reset', HoverTool(tooltips=[("Politician", "@x"), ("Similarity", "@top")])]


    p = figure(x_range=matrix.index.to_list(), plot_height=400,
               plot_width=600, tools=tools)


    p.vbar(x=matrix.index.to_list(), top=matrix.values.flatten(), width=0.75)

    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Similarity'
    p.xaxis.major_label_orientation = 45
    p.xaxis.major_label_text_font_size = "10pt"
    p.axis.axis_label_text_font_size = "10pt"

    return(p)


def determine_user_similarity(username, reference_matrix):
    matrix_copy = reference_matrix.copy()

    #scrape profile of entered user
    user_df = scrape_user(username)

    #clean tweets for sentiment analysis
    user_df['tweet'] = user_df['tweet'].apply(lambda x: clean_tweet_sentiment_analysis(x))

    #perform sentiment analysis on user's tweets
    sentiment_analyzer = SentimentIntensityAnalyzer()
    user_df['sentiment'] = user_df['tweet'].apply(lambda x: calc_tweet_sentiment(x, sentiment_analyzer))

    #prepare tweets for tf-idf vectorization
    user_df['cleaned_tweets'] = user_df['tweet'].apply(lambda x: clean_tweet_tfidf(x))
    user_df['lemmatized_tweets'] = lemmatize_tweets(user_df['cleaned_tweets'])

    #determine word vectors for entered user
    user_word_vector = calc_word_vectors(user_df, reference_matrix.columns.to_list())

    matrix_copy.loc[username] = user_word_vector

    cos_sim_df = create_similarity_matrix(cosine_similarity, matrix_copy)

    return(matrix_copy, cos_sim_df)
