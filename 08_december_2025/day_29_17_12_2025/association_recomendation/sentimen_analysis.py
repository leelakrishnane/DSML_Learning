import nltk
import random
import pandas as pd

from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('movie_reviews')

documents = []
labels = []

# Load text and sentiment labels
for fileid in movie_reviews.fileids():
    documents.append(movie_reviews.raw(fileid))
    labels.append(movie_reviews.categories(fileid)[0])

# Create DataFrame
df = pd.DataFrame({
    "review": documents,
    "sentiment": labels
})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.head()


vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df['review'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert to DataFrame for readability
cosine_sim_df = pd.DataFrame(cosine_sim)

cosine_sim_df.head()

def get_similar_reviews(index, top_n=5):
    similarity_scores = list(enumerate(cosine_sim[index]))
    
    # Sort by similarity score (excluding itself)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    results = []
    for i, score in similarity_scores:
        results.append({
            "review_index": i,
            "similarity_score": score,
            "sentiment": df.loc[i, 'sentiment'],
            "review_snippet": df.loc[i, 'review'][:150] + "..."
        })
    
    return pd.DataFrame(results)


# Pick a review to compare
review_index = 0

print("Original Review Sentiment:", df.loc[review_index, 'sentiment'])
print(df.loc[review_index, 'review'][:300], "\n")

similar_reviews = get_similar_reviews(review_index, top_n=5)
similar_reviews
