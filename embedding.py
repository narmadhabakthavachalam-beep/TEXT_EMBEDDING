import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Read CSV file
df = pd.read_csv('data/text_generation_results.csv')

print(df.head())
print("Number of Texts:", len(df))

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get texts
texts = df['Generated_Text'].astype(str).tolist()

# Create embeddings
embeddings = model.encode(texts)

print("Embeddings shape:", embeddings.shape)

# Save embeddings
np.save('data/embeddings.npy', embeddings)

print("Embedding file created successfully.")

# Load embeddings
embeddings = np.load('data/embeddings.npy')

print("Loaded embeddings shape:", embeddings.shape)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)

# Compare every text with every other text
results = []

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        results.append({
            'Text 1': texts[i],
            'Text 2': texts[j],
            'Cosine Similarity': round(similarity_matrix[i][j], 4)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by similarity
results_df = results_df.sort_values(
    by='Cosine Similarity',
    ascending=False
)

# Save results
results_df.to_csv(
    'data/similarity_results.csv',
    index=False
)

print("Comparison completed")

print("\nTop 5 most similar pairs:")
print(results_df.head(5))