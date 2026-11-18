
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, retrieved_chunks):

    if not retrieved_chunks:
        return []

    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    scores = model.predict(pairs)

    ranked_chunks = []

    for chunk, score in zip(retrieved_chunks, scores):

        ranked_chunks.append({
            **chunk,
            "rerank_score": float(score)
        })

    ranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_chunks
