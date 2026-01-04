import snscrape.modules.twitter as sntwitter
import pandas as pd
query = "climate change lang:en"

# Search keyword
query = "climate change"

# Number of tweets to scrape
limit = 100

tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    tweets.append([
        tweet.date,
        tweet.user.username,
        tweet.content,
        tweet.likeCount,
        tweet.retweetCount
    ])

# Create DataFrame
df = pd.DataFrame(tweets, columns=[
    "Date", "Username", "Tweet", "Likes", "Retweets"
])

# Save to CSV
df.to_csv("data/tweets.csv", index=False)

print("Tweets scraped successfully!")
