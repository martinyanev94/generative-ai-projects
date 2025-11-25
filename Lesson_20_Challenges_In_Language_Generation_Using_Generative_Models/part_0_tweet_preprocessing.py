import re
import pandas as pd

def preprocess_tweets(tweets):
    # Remove URLs and mentions
    tweets = [re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE) for tweet in tweets]
    tweets = [re.sub(r'@\w+', '', tweet) for tweet in tweets]
    # Tokenize the tweets
    tweets = [tweet.split() for tweet in tweets]
    return tweets

# Example: Consider we have a pandas dataframe
data = {'tweets': ["Check out this link http://example.com", "@user1 Great news!"]}
df = pd.DataFrame(data)

# Preprocessing the tweets
preprocessed_tweets = preprocess_tweets(df['tweets'])
print(preprocessed_tweets)
