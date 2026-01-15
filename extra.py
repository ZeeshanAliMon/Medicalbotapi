import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# load API key
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# init client
pc = Pinecone(api_key=PINECONE_API_KEY)

# connect to your index
index = pc.Index("medicalbot")

# embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# embed query
query_text = "What are symptoms of diabetes?"
query_vector = embedder.encode(query_text).tolist()

# query Pinecone
results = index.query(
    vector=query_vector,
    top_k=4,
    include_metadata=True
)

for match in results["matches"]:
    print(match["metadata"].get("text"))
