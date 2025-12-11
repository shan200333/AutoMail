import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# ---------------------------
# STEP 1 — Load Your Dataset
# ---------------------------

emails = pd.read_csv("emails.csv")
replies = pd.read_csv("replies.csv")

n_emails = len(emails)
n_replies = len(replies)

print("Emails loaded:", n_emails)
print("Replies loaded:", n_replies)

# -----------------------------------------------
# STEP 2 — Generate Bipartite Graph (52% density)
# -----------------------------------------------

density = 0.52
total_pairs = n_emails * n_replies
num_edges = int(total_pairs * density)

print("Total possible edges:", total_pairs)
print("Edges required for 52% density:", num_edges)

edge_indices = np.random.choice(total_pairs, size=num_edges, replace=False)

email_ids = edge_indices // n_replies + 1
reply_ids = edge_indices % n_replies + 1

bipartite_edges = pd.DataFrame({
    "Email_ID": email_ids,
    "Reply_ID": reply_ids
})

bipartite_edges.to_csv("bipartite_edges.csv", index=False)
print("Saved: bipartite_edges.csv")

# ---------------------------------------------
# STEP 3 — TEXT EMBEDDINGS (Subject + Body)
# ---------------------------------------------

print("Loading embedding model…")
model = SentenceTransformer("all-MiniLM-L6-v2")   # fast & powerful

# Combine fields for better semantic embedding
emails["FullText"] = emails["Subject"].astype(str) + " " + emails["Body"].astype(str)
replies["FullText"] = replies["Reply_Subject"].astype(str) + " " + replies["Reply_Body"].astype(str)

print("Generating embeddings for emails…")
email_embeddings = model.encode(
    emails["FullText"].tolist(),
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Generating embeddings for replies…")
reply_embeddings = model.encode(
    replies["FullText"].tolist(),
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Save embeddings as .npy
np.save("email_embeddings.npy", email_embeddings)
np.save("reply_embeddings.npy", reply_embeddings)

print("Saved: email_embeddings.npy")
print("Saved: reply_embeddings.npy")

# ---------------------------------------------
# STEP 4 — Final Output Summary
# ---------------------------------------------
print("\n============================")
print(" ALL FILES GENERATED SUCCESSFULLY")
print("============================")
print("• emails.csv")
print("• replies.csv")
print("• bipartite_edges.csv  (12M+ edges)")
print("• email_embeddings.npy (3977 × 384)")
print("• reply_embeddings.npy (6023 × 384)")
print("============================")
