# Local Politician Twitter Recommender
This application was created by [Matt Sternke](https://www.linkedin.com/in/matt-sternke/) as a capstone project for [The Data Incubator](https://www.thedataincubator.com/).

Voter turnout in local political elections is [incredibly low](http://whovotesformayor.org/). While this is a very nuanced problem, one major contributing factor is that [voters are unfamiliar with local political candidates](https://hub.jhu.edu/2018/12/14/americans-dont-understand-state-government/).

To help address this problem, I built a web application that helps interested voters compare and determine similarities between local politicians and any politician that they recognize and know more about.

# Comparing politicians using Twitter
These days, nearly all politicians use Twitter as a political platform to directly connect with voters and broadcast their political ideologies. Thus, a politician's Twitter profile represents an effective means to compare politicians.

To analyze the politicians, I scraped the most recent Tweets for a number of politicians. I apply cleaning and pre-processing steps to prepare the Tweets for analysis. To analyze the politician's Twitter profile, I use techniques of [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) to identify the most significant words or ideas Tweeted by each politicians, then analyze the sentiment of the Tweets containing these important words or ideas. The similarities between politicians are then quantified using a cosine similarity metric over all word vectors for each politician.

# Website
The website is a Flask application that will be deployed on Heroku at: {insert address here}. Alternatively, the app can be run locally by following the given instructions:
1. `git clone` this repository
2. `pip install -r requirements` to install all necesssary dependencies
3. `python app.py` to run the application
4. Navigate to `localhost:5000` in a web browser to explore!

# What can I do on the app?
Currently, the application will allow users to compare the current registered candidates for the 2021 NYC mayoral election to any person of interest. Users will enter the Twitter handle of a particular politician they want to use as comparison. Upon submitting, the app will scrape the entered handle's most recent Tweets, run all the steps for analysis, determine the similarities to all candidates, and display the similarities in a chart.
