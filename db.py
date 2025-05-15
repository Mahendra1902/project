# build_vectordb.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Sample historical incident logs
incident_logs = [
    "Worker slipped near chemical storage area.",
    "Gas leak detected in maintenance section.",
    "Unauthorized access to furnace area.",
    "No helmet detected near Furnace 3.",
    "High noise levels in turbine room."
]

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(incident_logs)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and logs
faiss.write_index(index, "incident_faiss.index")
with open("incident_logs.pkl", "wb") as f:
    pickle.dump(incident_logs, f)

print("Vector DB created and saved.")