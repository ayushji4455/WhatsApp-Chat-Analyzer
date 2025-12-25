from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# --- IMPROVEMENTS ---
# 1. Load extractor and stop words only ONCE at the module level
extract = URLExtract()
with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
    # 2. Use a set for much faster "in" checks
    stop_words = set(f.read().splitlines())
# --------------------

def fetch_stats(df):
    # Fetch the total number of messages
    num_messages = df.shape[0]

    # Fetch the total number of words
    words = []
    # Add a check to ensure 'message' is a string before splitting
    for message in df['message']:
        if isinstance(message, str):
            words.extend(message.split())
    num_words = len(words)

    # Fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Fetch number of links shared
    links = []
    # Add a check here as well
    for message in df['message']:
        if isinstance(message, str):
            links.extend(extract.find_urls(message))
    num_links = len(links)

    return num_messages, num_words, num_media_messages, num_links


def most_busy_users(df):
    x = df['user'].value_counts().head()
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'name', 'count': 'percent'})
    return x, percent_df

def create_wordcloud(df):
    # Filter out non-message rows
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    # Function to remove stop words from a single message
    def remove_stop_words(message):
        y = [word for word in message.lower().split() if word not in stop_words]
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(df):
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(df):
    emojis = []
    # Add a check to ensure 'message' is a string before iterating
    for message in df['message']:
        if isinstance(message, str):
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(df):
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    # (IMPROVEMENT) Create the 'time' column efficiently without a loop
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

def daily_timeline(df):
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(df):
    return df['day_name'].value_counts()

def month_activity_map(df):
    return df['month'].value_counts()

def activity_heatmap(df):
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd  # Make sure pandas is imported


def sentiment_analysis(df):
    # Create an instance of the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Create new columns for sentiment scores
    sentiment_scores = df['message'].apply(lambda msg: analyzer.polarity_scores(str(msg)))
    df['sentiment_scores'] = sentiment_scores

    # Extract the compound score, which is a single, useful metric
    df['sentiment_compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

    # Create a sentiment label based on the compound score
    def get_sentiment_label(compound_score):
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment_label'] = df['sentiment_compound'].apply(get_sentiment_label)

    return df


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
import pandas as pd  # Make sure pandas is imported


# ... keep the STOP_WORDS_SET defined at the top of your file

def find_topics(df, num_topics=5, num_words=10):
    """
    Finds the main topics in the chat using LDA.
    """
    # 1. Filter and prepare the text data
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')].copy()

    # 2. Lemmatize the messages (reduce words to their root form)
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(message):
        tokens = nltk.word_tokenize(message.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        return " ".join(lemmatized_tokens)

    temp['lemmatized'] = temp['message'].apply(lemmatize_text)

    # 3. Vectorize the text data (convert text to a matrix of word counts)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(temp['lemmatized'])

    # 4. Create and fit the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # 5. Get the top words for each topic
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))

    return topics


