from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = ["The bat flew out of the cave", "I hit the ball with my bat"]
embeddings = model.encode(sentences)

# Compute similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Cosine similarity: {similarity[0][0]}")
