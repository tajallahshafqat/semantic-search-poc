import torch
import json

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

f = open("data/jobs.json")
data = json.load(f)

position_titles = list(map(lambda job: job["job_position_title"], data["data"]))

corpus = []
[corpus.append(x) for x in position_titles if x not in corpus]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

queries = [
    "Senior Mining Estimator",
    "Mechanical Engineer",
    "Project Manager"
]

top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(corpus[idx], "(Score: {:.4f})".format(score))
