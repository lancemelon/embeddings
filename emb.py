import openai
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv

load_dotenv()

# Access the variables
embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=embeddings_api_key)

# Define words to visualize
words = ["man", "woman", "king", "queen"]

# Function to get embeddings from OpenAI
def get_embeddings(words):
    # Use the new client method for embeddings
    response = client.embeddings.create(
        model="text-embedding-ada-002",  # Specify the embeddings model
        input=words                       # Pass the words as input
    )
    # Extract the embeddings correctly
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

# Get the embeddings for these words
vectors = get_embeddings(words)

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Plot the words
plt.figure(figsize=(8, 6))
for word, coord in zip(words, reduced_vectors):
    plt.scatter(*coord, label=word)
    plt.text(coord[0] + 0.02, coord[1] + 0.02, word, fontsize=12)

plt.legend()
plt.title("Word Embeddings Visualization for 'man', 'woman', 'king', 'queen'")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()
