# Local Politician Twitter Recommender

This application was created by [Matt Sternke](https://www.linkedin.com/in/matt-sternke/) as a capstone project for [The Data Incubator](https://www.thedataincubator.com/).

Voter turnout in local political elections is [incredibly low](http://whovotesformayor.org/). While this is a very nuanced problem, one major contributing factor is that [voters are unfamiliar with local political candidates](https://hub.jhu.edu/2018/12/14/americans-dont-understand-state-government/).

To help address this problem, I built a web application that helps interested voters compare and determine similarities between local politicians and any politician that they recognize and know more about.

# Comparing politicians using Twitter
These days, nearly all politicians use Twitter as a political platform to directly connect with voters and broadcast their political ideologies. Thus, a politician's Twitter profile represents an effective means to compare politicians.

To analyze the politicians, I scraped the most recent Tweets for a number of politicians. I apply cleaning and pre-processing steps to prepare the Tweets for analysis. To analyze the politician's Twitter profile, I use techniques of [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) to identify the most significant words or ideas Tweeted by each politicians, then analyze the sentiment of the Tweets containing these important words or ideas. The similarities between politicians are then quantified using a cosine similarity metric over all word vectors for each politician.
