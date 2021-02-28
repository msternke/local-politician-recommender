import pandas as pd
import pickle
import os
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
from bokeh.models import ColumnDataSource, HoverTool, Legend

from wordcloud import WordCloud, STOPWORDS

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
    '''
    Scrapes tweets for the user entered politicians

    Parameters:
        username (str): Twitter handle of politician entered by user

    Returns:
        tweets_df_filtered (pandsd dataframe): dataframe of scraped tweets

    '''

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
    '''
    Cleans tweet for sentiment analysis

    Parameters:
        tweet (str): Tweet to be cleaned

    Returns:
        tweet (str): Cleaned tweet

    '''

    #converts html to text
    tweet = BeautifulSoup(tweet, 'lxml').text

    #removes url links
    tweet = re.sub(r'http\S+', '', tweet)

    #removes twitter usernames
    tweet = re.sub(r'(\s)?@\w+', '', tweet)

    return(tweet)

def calc_tweet_sentiment(tweet, sent_analyzer):
    '''
    Calculates sentiment score of Tweet

    Parameters:
        tweet (str): Tweet to Analyze
        sent_analyser (VaderSentiment instance): sentiment analyzer object

    Returns:
        compound_score (float): Compound score of tweet; sentiment score
    '''

    compound_score = sent_analyzer.polarity_scores(tweet)['compound']

    return compound_score

def clean_tweet_tfidf(tweet):
    '''
    Cleans tweets for TF-IDF

    Parameters:
        tweet (str): Tweet to clean

    Returns:
        cleaned_tweet (str): Cleaned Tweet
    '''

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
    cleaned_tweet = ' '.join([item for item in cleaned_tweet.split() \
        if item not in custom_stop_words])

    return(cleaned_tweet)

def get_wordnet_pos(word):
    '''
    Maps POS tag to word as character Lemmatize function accepts

    Parameters:
        word (str): Word to tags

    Returns:
        tagged_char (str): Character for tag
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    tagged_char = tag_dict.get(tag, wordnet.NOUN)

    return tagged_char

def lemmatize_tweets(tweets):
    '''
    Lemmatize Tweets for TF-IDF analysis

    Parameters:
        tweets (list): List of tweets to lemmatize

    Returns:
        lemmatized_list (list): List of lemmatized Tweets
    '''

    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    #lemmatize the words
    lemmatized_list = [' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) \
        for w in nltk.word_tokenize(tweet)]) for tweet in tweets]

    return lemmatized_list

def calc_word_vectors(df, words):
    '''
    Calculate vectors of top words as mean sentiment scores of tweets containing
    a given word

    Parameters:
        df (dataframe): Politician dataframe containing Tweets
        words(list): List of top words among politicians

    Returns:
        word_scores (list): List scores for each word to make vector
    '''

    word_scores = []
    for word in words:
        word_df = df[df['lemmatized_tweets'].str.contains(word)]
        if word_df.empty:
            word_scores.append(0)
        else:
            word_scores.append(word_df['sentiment'].mean())
    return(word_scores)

def create_similarity_matrix(similarity_tool, df_vectors):
    '''
    Creates similarity matrix from word vectors among group of politicians

    Parameters:
        similarity_tool (obj): Tool used to determine similarities
        df_vectors (dataframe): matrix of word vectors for each politician

    Returns:
        df (dataframe): Matrix of similarity scores as dataframe
    '''

    similarities = np.zeros((len(df_vectors), len(df_vectors)))

    for i in range(len(df_vectors)):
        politician_1 = df_vectors.iloc[i,:].values.reshape(1, -1)
        for j in range(i, len(df_vectors)):
            politician_2 = df_vectors.iloc[j, :].values.reshape(1, -1)
            similarities[i][j] = similarities[j][i] = similarity_tool(politician_1,politician_2)

    df = pd.DataFrame(similarities, index = df_vectors.index, columns = df_vectors.index)

    return df

def make_sim_bar_chart(matrix, info_df):
    '''
    Makes bar chart showing similarity scores to user entered politician

    Parameters:
        matrix (dataframe): Matrix of similarity scores
        info_df (dataframe): Dataframe containing information on politicians

    Returns:
        p (figure): Bokeh figure of bar chart
    '''

    merged = matrix.merge(info_df, left_index = True, right_on='name')

    color_dict = {'Democrat': 'royalblue', 'Republican': 'firebrick', 'Other': 'gray'}
    colors = [color_dict[party] for party in merged['party']]

    source = ColumnDataSource(data={
        'x': [i.replace("-"," ") for i in matrix.index],
        'top': matrix.values.flatten(),
        'color': colors,
        'label': merged['party'].to_list(),
        'width': [0.9]*len(merged)})

    p = figure(x_range=matrix.index.to_list(), plot_height=500, plot_width=700, toolbar_location=None)
    p.add_layout(Legend(), 'right')

    p.vbar(x='x', top='top', width='width', color='color', legend_field='label', source=source)

    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Similarity'
    p.xaxis.major_label_orientation = 45
    p.xaxis.major_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_size = "12pt"
    p.add_tools(HoverTool(tooltips=[("Politician", "@x"), ("Similarity", "@top")]))

    return(p)

def determine_user_similarity(username, reference_matrix):
    '''
    Main function to run analysis for determining similarity of user entered
    politician to set of reference politicians

    Parameters:
        username (str): Twitter handle of politician entered by user
        reference_matrix (dataframe): Matrix of reference politician word vectors

    Returns:
        matrix_copy (dataframe): Matrix of word vectors with user entered politician added
        cos_sim_df (dataframe): Matrix of politician similarities
    '''

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
