import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


all_tweets = pd.read_json("random_tweets.json", lines=True)

median_retweets = all_tweets["retweet_count"].median()

all_tweets["is_viral"] = np.where(all_tweets["retweet_count"] > median_retweets, 1, 0)

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)

all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']["followers_count"], axis=1)

all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']["friends_count"], axis=1)

all_tweets['hashtag_count'] = all_tweets.apply(lambda tweet: tweet['text'].count("#"), axis=1)

labels = all_tweets['is_viral']

data = all_tweets[['tweet_length', 'followers_count', 'friends_count', 'hashtag_count']]

scaled_data = scale(data, axis=0)

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size=0.2, random_state=1)

scores = []
for k in range(1, 200):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    scores.append(knn.score(test_data, test_labels))

plt.plot(scores)
plt.show()