from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ---------------------------
# Load and preprocess data
# ---------------------------
merged_df = pd.read_csv("merged_df.csv")

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(merged_df['content'])

# User-item encoding
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
merged_df['user_enc'] = user_encoder.fit_transform(merged_df['user_id'])
merged_df['news_enc'] = item_encoder.fit_transform(merged_df['news_id'])

user_item_matrix = merged_df.pivot_table(index='user_enc', columns='news_enc', values='clicked', fill_value=0)
user_similarity = cosine_similarity(user_item_matrix)
merged_df = merged_df.reset_index(drop=True)

# ---------------------------
# Recommendation Functions
# ---------------------------
def content_base_rec(title, top_n=5):
    # cleaning
    title = re.sub(r'\W+', ' ', title)  # Remove special chars
    title = title.lower().strip()
    # Vectorize input title
    title_vec = vectorizer.transform([title])
    # Compute cosine similarity
    sim_scores = cosine_similarity(title_vec, tfidf_matrix).flatten()
    # Get top N indices
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return merged_df.loc[top_indices, ['news_id', 'title', 'category', 'subcategory', 'url', 'abstract']]


def collaborative_base_rec(input_user, df, top_k=5):
    # Encode user_id to index
    user_idx = user_encoder.transform([input_user])[0]
    # Get similarity scores for this user with others
    sim_scores = user_similarity[user_idx]
    # Get clicked items by this user
    user_clicks = user_item_matrix.iloc[user_idx]
    # Calculate weighted sum of clicks by similarity scores
    weighted_scores = sim_scores @ user_item_matrix.values
    # Remove already clicked items from recommendations
    weighted_scores[user_clicks == 1] = 0
    # Get top news indices based on scores
    top_news_indices = weighted_scores.argsort()[::-1][:top_k]
    # Decode back to news_ids
    recommended_news_ids = item_encoder.inverse_transform(top_news_indices)
    # Filter original df for those news and drop duplicates
    recommended_news = df[df['news_id'].isin(recommended_news_ids)][
        ['news_id', 'title', 'category', 'subcategory', 'url', 'abstract']
    ].drop_duplicates(subset='news_id')

    return recommended_news.reset_index(drop=True)


def hybrid_recommendations(user_id, title, df, top_n=5):
    # clean title
    title = re.sub(r'\W+', ' ', title)  # Remove special chars
    title = title.lower().strip()
    # Get content-based recommendations
    content_recs = content_base_rec(title)
    # Get collaborative-based recommendations
    collab_recs = collaborative_base_rec(user_id, df)
    # Merge the recommendations
    combined_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().reset_index(drop=True)
    # Limit the number of recommendations to the top 'n'
    combined_recs = combined_recs.head(top_n)
    return combined_recs

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        user_id = request.form['user_id']
        title = request.form['title']
        recommendations = hybrid_recommendations(user_id, title, merged_df).to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
