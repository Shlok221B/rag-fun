
import re
from rank_bm25 import BM25Okapi
from vector_store import collection
from embeddings import generate_embeddings


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def keyword_search(query, n_results=5):

    records = collection.get(
        where={"document_type": "employee_handbook"},
        include=["documents", "metadatas"]
    )

    documents = records["documents"]
    ids = records["ids"]
    metadatas = records["metadatas"]

    if not documents:
        return []

    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        {
            "id": ids[i],
            "text": documents[i],
            "score": float(scores[i]),
            "metadata": metadatas[i] or {}
        }
        for i in ranked_indices[:n_results]
        if scores[i] > 0
    ]


def hybrid_search(query, n_results=5, k=60):

    # 1. Vector search
    vector_results = collection.query(
        query_embeddings=generate_embeddings([query]).tolist(),
        n_results=n_results,
        where={"document_type": "employee_handbook"},
        include=["documents", "metadatas"]
    )

    # 2. BM25 search
    bm25_results = keyword_search(query, n_results)

    # 3. Reciprocal Rank Fusion
    scores = {}
    documents = {}
    metadata_map = {}

    for rank, doc_id in enumerate(vector_results["ids"][0], start=1):

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        documents[doc_id] = vector_results["documents"][0][rank - 1]

        metadata_map[doc_id] = (
            vector_results["metadatas"][0][rank - 1] or {}
        )

    for rank, result in enumerate(bm25_results, start=1):

        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        documents[doc_id] = result["text"]

        metadata_map[doc_id] = result["metadata"]

    # 4. Sort by combined score
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # 5. Return chunks with metadata
    return [
        {
            "id": doc_id,
            "rrf_score": score,
            "text": documents[doc_id],
            "metadata": metadata_map.get(doc_id, {})
        }
        for doc_id, score in ranked[:n_results]
    ]


if __name__ == "__main__":

    query = "What is the notice period for employees?"

    results = hybrid_search(query)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("RRF Score:", result["rrf_score"])
        print("Chunk ID:", result["id"])
        print("Metadata:", result["metadata"])
        print("Content:", result["text"])
