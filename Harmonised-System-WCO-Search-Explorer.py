# Theme Configuration
import streamlit as st

st.set_page_config(
    page_title="WCO HS 2022 Explorer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
    theme={
        "base": "light",
        "primaryColor": "#0072ce",  # WCO Blue
        "backgroundColor": "#d9d9d9",  # WCO Grey
        "textColor": "#0072cd"
    }
)

import pandas as pd
import os
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data Loading
repo_url = "https://github.com/datasets/harmonized-system.git"
repo_name = "harmonized-system"

if not os.path.exists(repo_name):
    os.system(f"git clone {repo_url}")

data_dir = f"./{repo_name}/data"
file_name = 'harmonized-system.csv'
file_path = os.path.join(data_dir, file_name)

try:
    df = pd.read_csv(file_path, low_memory=True)
    st.write("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 2. Data Cleaning and Preprocessing
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    return ""

if 'description' in df.columns:
    df['cleaned_description'] = df['description'].apply(clean_text)

if 'section' in df.columns:
    section_counts = df['section'].value_counts()
    top_20_sections = section_counts.index[:20]
    df_top20 = df[df['section'].isin(top_20_sections)]

if 'description' in df.columns:
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)

# 3. Train Relevance Prediction Model (dummy data)
df['relevance'] = (df['sentiment'] + 1) / 2  # Dummy relevance
X = df[['sentiment']]
y = df['relevance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- Streamlit Search Functionality ---

def search_descriptions(search_term):
    search_term = search_term.lower()

    df['combined_text'] = df['cleaned_description'].astype(str) + ' ' + \
                           df['hscode'].astype(str) + ' ' + \
                           df['section'].astype(str)

    if search_term.isdigit():
        results = df[df['hscode'].astype(str).str.contains(search_term, na=False, regex=False)].copy()
        results['predicted_relevance'] = 1.0  # Default relevance for HS code search
    else:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
        search_vector = vectorizer.transform([search_term])

        cosine_similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()
        df['relevance'] = cosine_similarities

        results = df[df['combined_text'].str.contains(search_term, na=False, regex=False)].copy()

        if not results.empty:
            results['predicted_relevance'] = model.predict(results[['sentiment']])
            results = results.sort_values(by='predicted_relevance', ascending=False)

    if results.empty:
        st.write(f"No results found for '{search_term}'.")
    else:
        if 'predicted_relevance' in results.columns:
            results['relevance_percent'] = (results['predicted_relevance'] / results['predicted_relevance'].max()) * 100
            st.dataframe(results[['hscode', 'description', 'section', 'relevance_percent']])
        else:
            st.write("No predicted relevance found for the search results.")

# --- Streamlit App ---
st.title("WCO HS 2022 Explorer")

search_term = st.text_input("Enter HS code (4 digits) or words")

if st.button("Search"):
    if search_term:
        search_descriptions(search_term)

# Custom CSS
st.markdown("""
<style>.title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">WCO HS 2022 Explorer</div>', unsafe_allow_html=True)
