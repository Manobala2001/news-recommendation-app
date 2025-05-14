import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
from datetime import datetime
from googletrans import Translator
from transformers import pipeline

# Set a custom theme
st.set_page_config(page_title="Personalized News Bot", layout="wide", page_icon="📰")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
translator = Translator()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("english_news_dataset.csv")
    df.dropna(subset=['Content'], inplace=True)
    df['Content'] = df['Content'].astype(str)
    return df

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Translate content if needed
def translate_text(text, dest_lang):
    if dest_lang != 'en':
        try:
            translated = translator.translate(text, dest=dest_lang)
            return translated.text
        except:
            return text  # fallback if translation fails
    return text

# Generate summary for a news article
def generate_summary(text):
    try:
        summary = summarizer(text[:1024], max_length=80, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except:
        return text[:250]  # fallback to truncated content

df=df.drop_duplicates(subset=['Content','Headline'])
df=df.reset_index(drop=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'liked_articles' not in st.session_state:
    st.session_state.liked_articles = set()
if 'disliked_articles' not in st.session_state:
    st.session_state.disliked_articles = set()
if 'bookmarked_articles' not in st.session_state:
    st.session_state.bookmarked_articles = set()
if 'username' not in st.session_state:
    st.session_state.username = None

# User Login Simulation
st.sidebar.title("🔐 Login")
username_input = st.sidebar.text_input("Enter username")
if st.sidebar.button("Login"):
    st.session_state.username = username_input

if not st.session_state.username:
    st.warning("Please login from the sidebar to continue.")
    st.stop()

# Language selector
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    "Arabic": "ar"
}
selected_language = st.sidebar.selectbox("🌐 Select Language", list(language_map.keys()))
dest_lang = language_map[selected_language]

st.title(f"📰 Welcome, {st.session_state.username} - Your Personalized News Bot")
user_query = st.text_input("🔍 What's on your mind today?")
data = load_data()

# Optional category filter
all_categories = sorted(set(eval(cat)[0] for cat in data['News Categories'].unique()))
selected_category = st.selectbox("📂 Filter by Category (optional):", ["All"] + all_categories)
if selected_category != "All":
    data = data[data['News Categories'].str.contains(selected_category)]

# Date range filter
min_date = pd.to_datetime(data['Date'], errors='coerce').min()
max_date = pd.to_datetime(data['Date'], errors='coerce').max()
start_date, end_date = st.date_input("📅 Filter by Date Range:", [min_date, max_date])
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
# Convert start_date and end_date to pandas Timestamp objects for comparison
data = data[(data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))]

# Define color palette for categories
category_colors = {
    category: color for category, color in zip(
        all_categories,
        ["#F94144", "#F3722C", "#F8961E", "#F9C74F", "#90BE6D", "#43AA8B", "#577590"] * 10
    )
}

# Load and encode model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

@st.cache_data
def encode_corpus(corpus):
    return model.encode(corpus, show_progress_bar=True)
embeddings = encode_corpus(data['Content'].tolist())

# Topic Clustering
@st.cache_data
def cluster_topics(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)
data['Topic Cluster'] = cluster_topics(embeddings)

if user_query:
    query_embedding = model.encode([user_query])[0]
    st.session_state.query_history.append(query_embedding)
    user_profile = np.mean(st.session_state.query_history, axis=0)

    similarity_scores = cosine_similarity([user_profile], embeddings).flatten()
    top_indices = similarity_scores.argsort()[::-1]

    st.subheader("📌 Recommended News Articles:")
    shown = 0
    for i in top_indices:
        if shown >= 5:
            break
        if i in st.session_state.liked_articles or i in st.session_state.disliked_articles:
            continue

        with st.container():
            category = eval(data.iloc[i]['News Categories'])[0]
            color = category_colors.get(category, "#CCCCCC")
            summary = generate_summary(data.iloc[i]['Content'])
            translated_summary = translate_text(summary, dest_lang)
            translated_headline = translate_text(data.iloc[i]['Headline'], dest_lang)
            st.markdown(f"**🗞️ {translated_headline}**")
            st.markdown(f"{translated_summary}...")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>📂 {category}</span> | _Date: {data.iloc[i]['Date'].date()}_", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(f"👍 Like {i}"):
                    st.session_state.liked_articles.add(i)
            with col2:
                if st.button(f"👎 Dislike {i}"):
                    st.session_state.disliked_articles.add(i)
            with col3:
                if st.button(f"🔖 Bookmark {i}"):
                    st.session_state.bookmarked_articles.add(i)

            st.markdown(f"🔖 **Topic Cluster**: {data.iloc[i]['Topic Cluster']}")
            st.markdown("---")

        shown += 1

# Export liked articles
if st.sidebar.button("📥 Export Liked Articles"):
    liked_df = data.loc[list(st.session_state.liked_articles)]
    liked_articles = liked_df.to_dict(orient="records")
    filename = f"liked_articles_{st.session_state.username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(liked_articles, f, indent=4)
    st.sidebar.success(f"Exported to {filename}")

# Export bookmarked articles
if st.sidebar.button("🔖 Export Bookmarked Articles"):
    bookmarked_df = data.loc[list(st.session_state.bookmarked_articles)]
    bookmarked_articles = bookmarked_df.to_dict(orient="records")
    filename = f"bookmarked_articles_{st.session_state.username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(bookmarked_articles, f, indent=4)
    st.sidebar.success(f"Exported to {filename}")

# Reset user history
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.query_history = []
    st.session_state.liked_articles = set()
    st.session_state.disliked_articles = set()
    st.session_state.bookmarked_articles = set()
    st.success("Session cleared. Start fresh!")
